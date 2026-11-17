using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using K4os.Compression.LZ4;
using K4os.Compression.LZ4.Streams;
using nvDismDecode;
using rtxdisam;
using static RegressionTests.TestContext;
using static RegressionTests.TestHelpers;
using static RegressionTests.Fixtures;

namespace RegressionTests;

internal static class FatbinDecodingTests
{
    public static void Run()
    {
        // -- fatbin decoding ----------------------------------------------------------

        byte[] uncompressed = BuildFatbin(1, Encoding.ASCII.GetBytes("abcabcabcabc"), 0x41, 0);
        FatbinFile plain = FatbinFile.Parse(uncompressed);
        Check("fatbin plain entry count", plain.Entries.Count == 1);
        Check("fatbin plain kind", plain.Entries[0].IsPtx);
        Equal("fatbin plain payload",
            "abcabcabcabc", Encoding.ASCII.GetString(plain.Entries[0].Payload));

        FatbinEntry nvvmEntry = FatbinFile.Parse(
            BuildFatbin(8, Encoding.ASCII.GetBytes("lto-data"), 0x41, 0)).Entries[0];
        Check("fatbin nvvm kind", nvvmEntry.KindName == "nvvm" && nvvmEntry.IsLtoIr);
        Equal("fatbin nvvm payload",
            "lto-data", Encoding.ASCII.GetString(nvvmEntry.Payload));

        FatbinEntry tileEntry = FatbinFile.Parse(
            BuildFatbin(0x80, Encoding.ASCII.GetBytes("tile-data"), 0x41, 0)).Entries[0];
        Check("fatbin tileir kind", tileEntry.KindName == "tile ir" && tileEntry.IsTileIr);
        Equal("fatbin tileir payload",
            "tile-data", Encoding.ASCII.GetString(tileEntry.Payload));

        FatbinEntry identifierEntry = FatbinFile.Parse(
            BuildFatbin(1, Encoding.ASCII.GetBytes("id-data"), 0x41, 0,
                86, null, "my-ptx")).Entries[0];
        Equal("fatbin identifier", "my-ptx", identifierEntry.Identifier);

        FatbinEntry optionsEntry = FatbinFile.Parse(
            BuildFatbin(1, Encoding.ASCII.GetBytes("opt-data"), 0x41, 0,
                86, null, null, "-O3 -v")).Entries[0];
        Equal("fatbin ptxas options", "-O3 -v", optionsEntry.PtxasOptions);

        FatbinEntry bothEntry = FatbinFile.Parse(
            BuildFatbin(1, Encoding.ASCII.GetBytes("both-data"), 0x41, 0,
                86, null, "both-id", "-O3")).Entries[0];
        Equal("fatbin combined identifier", "both-id", bothEntry.Identifier);
        Equal("fatbin combined options", "-O3", bothEntry.PtxasOptions);

        FatbinEntry elfOptionsEntry = FatbinFile.Parse(
            BuildFatbin(2, Encoding.ASCII.GetBytes("elf-data"), 0x41, 0,
                86, null, "elf-id", "-O3")).Entries[0];
        Equal("fatbin elf ignores ptxas options", "",
            elfOptionsEntry.PtxasOptions);

        byte[] oldStyleFatbin = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(oldStyleFatbin, 0x1EE55A01);
        Check("old-style fatbin detected", FatbinFile.IsFatbin(oldStyleFatbin));
        Check("old-style empty fatbin parses",
            FatbinFile.Parse(oldStyleFatbin).Entries.Count == 0);

        byte[] embeddedOldCubin = BuildElfWithCuinfo(86, 0x6005604);
        byte[] oldStyleWithCubin = new byte[16 + embeddedOldCubin.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(oldStyleWithCubin, 0x1EE55A01);
        embeddedOldCubin.CopyTo(oldStyleWithCubin, 16);
        FatbinEntry oldCubinEntry = FatbinFile.Parse(oldStyleWithCubin).Entries[0];
        Check("old-style embedded cubin detected",
            oldCubinEntry.IsCubin && oldCubinEntry.Architecture == 86,
            $"arch = {oldCubinEntry.Architecture}");
        Check("old-style embedded cubin payload preserved",
            oldCubinEntry.Payload.AsSpan().SequenceEqual(embeddedOldCubin));

        string oldPtx = ".version 6.0\n.target sm_70\n.entry test() { ret; }\n";
        byte[] oldStyleWithPtx = new byte[16 + Encoding.ASCII.GetByteCount(oldPtx)];
        BinaryPrimitives.WriteUInt32LittleEndian(oldStyleWithPtx, 0x1EE55A01);
        Encoding.ASCII.GetBytes(oldPtx).CopyTo(oldStyleWithPtx, 16);
        FatbinEntry oldPtxEntry = FatbinFile.Parse(oldStyleWithPtx).Entries[0];
        Check("old-style embedded PTX detected",
            oldPtxEntry.IsPtx && oldPtxEntry.Architecture == 70,
            $"arch = {oldPtxEntry.Architecture}");
        Equal("old-style embedded PTX payload", oldPtx.TrimEnd('\n'),
            Encoding.ASCII.GetString(oldPtxEntry.Payload).TrimEnd('\n'));

        byte[] cuda102Ptx = Encoding.ASCII.GetBytes(
            ".version 6.4\n.target sm_75\n.entry test() { ret; }\n");
        byte[] cuda102Fatbin = BuildOldFatbin(
            1, cuda102Ptx, 0x41, 0, 75, "old-id", "-O3 -v");
        FatbinEntry cuda102Entry = FatbinFile.Parse(cuda102Fatbin).Entries[0];
        Equal("CUDA 10.2 PTX identifier", "old-id", cuda102Entry.Identifier);
        Equal("CUDA 10.2 PTX options", "-O3 -v", cuda102Entry.PtxasOptions);
        Check("CUDA 10.2 PTX architecture", cuda102Entry.Architecture == 75);
        Check("CUDA 10.2 PTX code version",
            cuda102Entry.CodeVersionMajor == 1 &&
            cuda102Entry.CodeVersionMinor == 7);

        byte[] lz4BlockOld = { 0x35, (byte)'a', (byte)'b', (byte)'c', 0x03, 0x00 };
        FatbinFile oldLz4Fatbin = FatbinFile.Parse(
            BuildOldFatbin(1, lz4BlockOld, 0x2041, 12, 75));
        Equal("CUDA 10.2 LZ4 payload",
            "abcabcabcabc", Encoding.ASCII.GetString(oldLz4Fatbin.Entries[0].Payload));
        Equal("CUDA 10.2 LZ4 name", "LZ4",
            oldLz4Fatbin.Entries[0].Compression);

        var oldStyleScanned = new List<byte[]>();
        FatbinScanner.Scan(
            oldStyleWithCubin, 0, oldStyleWithCubin.Length, oldStyleScanned);
        Check("old-style scanner finds container", oldStyleScanned.Count == 1);
        Check("old-style scanner payload parses",
            oldStyleScanned.Count == 1 &&
            FatbinFile.Parse(oldStyleScanned[0]).Entries.Count == 1);

        byte[] lz4Block = { 0x35, (byte)'a', (byte)'b', (byte)'c', 0x03, 0x00 };
        byte[] lz4 = BuildFatbin(1, lz4Block, 0x2041, 12);
        FatbinFile lz4Fatbin = FatbinFile.Parse(lz4);
        Equal("fatbin lz4 payload",
            "abcabcabcabc", Encoding.ASCII.GetString(lz4Fatbin.Entries[0].Payload));
        Equal("fatbin lz4 name", "LZ4", lz4Fatbin.Entries[0].Compression);

        byte[] lz4Source = new byte[20000];
        for (int i = 0; i < lz4Source.Length; i++)
            lz4Source[i] = (byte)(i % 13 + (i / 97) % 5);
        byte[] lz4Compressed = new byte[LZ4Codec.MaximumOutputSize(lz4Source.Length)];
        int lz4Length = LZ4Codec.Encode(
            lz4Source, 0, lz4Source.Length,
            lz4Compressed, 0, lz4Compressed.Length,
            LZ4Level.L09_HC);
        lz4Compressed = lz4Compressed[..lz4Length];
        FatbinFile lz4RoundTrip = FatbinFile.Parse(
            BuildFatbin(1, lz4Compressed, 0x2041, (ulong)lz4Source.Length));
        Check("fatbin lz4 block round-trip",
            lz4RoundTrip.Entries[0].Payload.AsSpan().SequenceEqual(lz4Source));

        byte[] zlibPayload;
        using (var compressed = new MemoryStream())
        {
            using (var zlib = new ZLibStream(compressed, CompressionLevel.SmallestSize, leaveOpen: true))
                zlib.Write(Encoding.ASCII.GetBytes("hello"));
            zlibPayload = compressed.ToArray();
        }
        FatbinFile zlibFatbin = FatbinFile.Parse(BuildFatbin(1, zlibPayload, 0x1041, 5));
        Equal("fatbin deflate payload",
            "hello", Encoding.ASCII.GetString(zlibFatbin.Entries[0].Payload));
        Equal("fatbin deflate name", "DEFLATE", zlibFatbin.Entries[0].Compression);

        FatbinFile paddedDeflate = FatbinFile.Parse(
            BuildFatbin(1, zlibPayload, 0x1041, 5, 86, 0));
        Equal("fatbin deflate padded payload",
            "hello", Encoding.ASCII.GetString(paddedDeflate.Entries[0].Payload));

        FatbinFile explicitDeflate = FatbinFile.Parse(
            BuildFatbin(1, zlibPayload, 0x1041, 5, 86, zlibPayload.Length));
        Equal("fatbin deflate explicit size payload",
            "hello", Encoding.ASCII.GetString(explicitDeflate.Entries[0].Payload));

        byte[] zstdPayload;
        using (var compressor = new ZstdSharp.Compressor())
            zstdPayload = compressor.Wrap(Encoding.ASCII.GetBytes("hello")).ToArray();
        FatbinFile zstdFatbin = FatbinFile.Parse(BuildFatbin(1, zstdPayload, 0x8041, 5));
        Equal("fatbin zstd payload",
            "hello", Encoding.ASCII.GetString(zstdFatbin.Entries[0].Payload));
        Equal("fatbin zstd name", "ZSTD", zstdFatbin.Entries[0].Compression);

        byte[] lz4FramePayload;
        using (var compressed = new MemoryStream())
        {
            using (var frame = LZ4Stream.Encode(compressed, leaveOpen: true))
                frame.Write(Encoding.ASCII.GetBytes("hello"));
            lz4FramePayload = compressed.ToArray();
        }
        FatbinFile lz4FrameFatbin = FatbinFile.Parse(
            BuildFatbin(1, lz4FramePayload, 0x4041, 5));
        Equal("fatbin lz4 frame payload",
            "hello", Encoding.ASCII.GetString(lz4FrameFatbin.Entries[0].Payload));
        Check("fatbin lz4 frame name",
            lz4FrameFatbin.Entries[0].Compression.StartsWith("LZ4v2"),
            lz4FrameFatbin.Entries[0].Compression);
    }
}
