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

internal static class HostContainerAndApiTests
{
    public static void Run()
    {
        // -- host container scanning --------------------------------------------------

        byte[] uncompressed = BuildFatbin(1, Encoding.ASCII.GetBytes("abcabcabcabc"), 0x41, 0);
        byte[] pe = BuildPe(uncompressed);
        var peBlobs = PeFatbinReader.FindEmbeddedFatbins(pe);
        Check("pe fatbin count", peBlobs.Count == 1);
        Check("pe fatbin parses", peBlobs.Count == 1 && FatbinFile.Parse(peBlobs[0]).Entries.Count == 1);

        byte[] peWrapper = BuildPeWithWrapper(uncompressed);
        var peWrapperBlobs = PeFatbinReader.FindEmbeddedFatbins(peWrapper);
        Check("pe wrapper count", peWrapperBlobs.Count == 1);
        Check("pe wrapper parses",
            peWrapperBlobs.Count == 1 &&
            FatbinFile.Parse(peWrapperBlobs[0]).Entries.Count == 1);

        byte[] coff = BuildCoff(uncompressed);
        var coffBlobs = CoffFatbinReader.FindEmbeddedFatbins(coff);
        Check("coff fatbin count", coffBlobs.Count == 1);
        Check("coff fatbin parses",
            coffBlobs.Count == 1 &&
            FatbinFile.Parse(coffBlobs[0]).Entries.Count == 1);

        byte[] elf = BuildElfWithFatbin(uncompressed);
        CubinFile cubin = CubinFile.Load(elf);
        Check("elf embedded fatbin count", cubin.EmbeddedFatbins.Count == 1);
        Check("elf embedded fatbin parses",
            cubin.EmbeddedFatbins.Count == 1 &&
            FatbinFile.Parse(cubin.EmbeddedFatbins[0]).Entries.Count == 1);

        byte[] renamedElf = BuildElfWithFatbin(uncompressed, ".rodata");
        CubinFile renamedCubin = CubinFile.Load(renamedElf);
        Check("elf renamed-section fallback", renamedCubin.EmbeddedFatbins.Count == 1);

        CubinFile cuinfo86 = CubinFile.Load(BuildElfWithCuinfo(86));
        Check("elf cuinfo arch 86", cuinfo86.Architecture == 86,
            $"arch = {cuinfo86.Architecture}");
        Check("elf cuinfo is device image", cuinfo86.IsDeviceImage);
        Check("elf cuinfo tool version",
            cuinfo86.Notes.CuinfoToolVersion == 0x86,
            $"version = {cuinfo86.Notes.CuinfoToolVersion}");
        CubinFile cuinfo100 = CubinFile.Load(BuildElfWithCuinfo(100));
        Check("elf cuinfo arch 100", cuinfo100.Architecture == 100,
            $"arch = {cuinfo100.Architecture}");
        CubinFile virtual75 = CubinFile.Load(BuildElfWithCuinfo(75, 0x5604));
        Check("elf flags override virtual arch", virtual75.Architecture == 86,
            $"arch = {virtual75.Architecture}");
        Check("elf flags keep virtual note", virtual75.Notes.Architecture == 75,
            $"virt = {virtual75.Notes.Architecture}");

        CubinFile tkinfoCubin = CubinFile.Load(BuildElfWithTkinfo());
        Check("elf tkinfo first note", tkinfoCubin.Notes.TkinfoPresent &&
            tkinfoCubin.Notes.TkinfoToolName == "nvlink" &&
            tkinfoCubin.Notes.TkinfoVersion == 2,
            $"tool = {tkinfoCubin.Notes.TkinfoToolName}");

        Check("host fatbin ELF is not device image",
            !CubinFile.Load(BuildElfWithFatbin(uncompressed)).IsDeviceImage);

        CubinFile relocCubin = CubinFile.Load(BuildRelocElf());
        Check("reloc elf section count", relocCubin.Sections.Count == 1,
            $"count = {relocCubin.Sections.Count}");
        CubinSection relocSection = relocCubin.Sections[0];
        Check("reloc count", relocSection.Relocations.Count == 3,
            $"count = {relocSection.Relocations.Count}");
        Check("reloc types",
            relocSection.Relocations[0].Type == 56 &&
            relocSection.Relocations[1].Type == 57 &&
            relocSection.Relocations[2].Type == 58);
        Check("reloc symbol names",
            relocSection.Relocations[0].SymbolName == "f" &&
            relocSection.Relocations[1].SymbolName == "f__0" &&
            relocSection.Relocations[2].SymbolName == "g");
        Check("reloc target dedup",
            relocSection.RelocationTargets.Count == 1 &&
            relocSection.RelocationTargets[0] == 0x20);
        Check("duplicate function suffix",
            relocSection.Functions.Count == 3 &&
            relocSection.Functions[0].Name == "f" &&
            relocSection.Functions[1].Name == "f__0" &&
            relocSection.Functions[1].OriginalName == "f" &&
            relocSection.Functions[2].Name == "g");

        CubinFile brxCubin = CubinFile.Load(BuildBrxElf());
        Check("brx info section parsed", brxCubin.Sections.Count == 1);
        CubinSection brxSection = brxCubin.Sections[0];
        Check("brx indirect targets",
            brxSection.IndirectBranchTargets.TryGetValue(
                0x10, out List<ulong>? brxTargets) &&
            brxTargets.Count == 3 &&
            brxTargets[0] == 0x100 &&
            brxTargets[1] == 0x120 &&
            brxTargets[2] == 0x140);
        Check("info stack offset attribute", brxSection.HasStackOffset);

        byte[] wrapperElf = BuildElfWithWrapper(uncompressed);
        CubinFile wrapperCubin = CubinFile.Load(wrapperElf);
        Check("elf .nvFatBinSegment wrapper", wrapperCubin.EmbeddedFatbins.Count == 1);
        Check("elf wrapper payload parses",
            wrapperCubin.EmbeddedFatbins.Count == 1 &&
            FatbinFile.Parse(wrapperCubin.EmbeddedFatbins[0]).Entries.Count == 1);

        string asmDir = Path.Combine(
            Path.GetTempPath(), "rtxdisasm-asm-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(asmDir);
        try
        {
            string sassPath = Path.Combine(asmDir, "sample.sass");
            string cubinPath = Path.Combine(asmDir, "sample.cubin");
            string irPath = Path.Combine(asmDir, "sample.ir.json");
            File.WriteAllText(sassPath, """
                .target sm_75
                .section .text.test,"ax",@progbits
                .global test
                .type test,@function
                test:
                    MOV R1, c[0x0][0x28];
                    EXIT;
                """);
            SassAssembler.Assemble(sassPath, "sm_75", cubinPath, irPath, "13.4");
            CubinFile assembled = CubinFile.Load(cubinPath);
            Check("assembler architecture", assembled.Architecture == 75,
                $"arch = {assembled.Architecture}");
            Check("sm75 cubin ABI", assembled.ToolkitFromAbi == "134");
            Check("sm75 CUDA 13.4 tkinfo",
                assembled.Notes.TkinfoToolVersion.Contains(
                    "release 13.4", StringComparison.Ordinal));
            string cubin10Path = Path.Combine(asmDir, "sample75-102.cubin");
            SassAssembler.Assemble(sassPath, "sm_75", cubin10Path, null, "10.2");
            CubinFile assembled10 = CubinFile.Load(cubin10Path);
            Check("sm75 CUDA 10.2 cubin ABI",
                assembled10.ToolkitFromAbi == "102");
            Check("sm75 CUDA 10.2 tkinfo",
                assembled10.Notes.TkinfoToolVersion.Contains(
                    "release 10.2", StringComparison.Ordinal));
            string inferredFatbinPath = Path.Combine(
                asmDir, "sample75-102-inferred.fatbin");
            FatbinWriter.Write(
                inferredFatbinPath, File.ReadAllBytes(cubin10Path),
                assembled10.Architecture);
            FatbinEntry inferredEntry = FatbinFile.Parse(
                File.ReadAllBytes(inferredFatbinPath)).Entries[0];
            Check("fatbin infers CUDA 10.x from cubin ABI",
                inferredEntry.CodeVersionMajor == 1 &&
                inferredEntry.CodeVersionMinor == 7);
            bool modernOnPreVoltaRejected = false;
            try
            {
                FatbinWriter.Write(
                    Path.Combine(asmDir, "modern-on-prevolta.fatbin"),
                    File.ReadAllBytes(cubinPath), 50);
            }
            catch (InvalidDataException)
            {
                modernOnPreVoltaRejected = true;
            }
            Check("fatbin rejects 13.x ABI on pre-Volta",
                modernOnPreVoltaRejected);
            bool legacyOnNewerRejected = false;
            try
            {
                FatbinWriter.Write(
                    Path.Combine(asmDir, "legacy-on-newer.fatbin"),
                    File.ReadAllBytes(cubin10Path), 80);
            }
            catch (InvalidDataException)
            {
                legacyOnNewerRejected = true;
            }
            Check("fatbin rejects 10.x ABI on SM80",
                legacyOnNewerRejected);
            string directWriterPath = Path.Combine(
                asmDir, "direct-writer-102.cubin");
            rtxdisam.CubinWriter.Write(
                directWriterPath, 75, "sm_75",
                new[]
                {
                    new rtxdisam.CubinImageSection
                    {
                        Name = ".text.k",
                        Alignment = 16,
                        Data = new byte[16],
                    },
                },
                new[]
                {
                    new rtxdisam.CubinImageSymbol
                    {
                        Name = "k",
                        Section = 0,
                        Value = 0,
                        Size = 16,
                    },
                },
                "cuda10.2");
            CubinFile directWriterCubin = CubinFile.Load(directWriterPath);
            Check("cubin writer normalizes toolkit",
                directWriterCubin.ToolkitFromAbi == "102" &&
                directWriterCubin.Notes.TkinfoToolVersion.Contains(
                    "release 10.2", StringComparison.Ordinal));
            string badDirectWriterPath = Path.Combine(
                asmDir, "direct-writer-80-102.cubin");
            bool directAbiRejected = false;
            try
            {
                rtxdisam.CubinWriter.Write(
                    badDirectWriterPath, 80, "sm_80",
                    Array.Empty<rtxdisam.CubinImageSection>(),
                    Array.Empty<rtxdisam.CubinImageSymbol>(), "10.2");
            }
            catch (InvalidDataException)
            {
                directAbiRejected = true;
            }
            Check("cubin writer rejects missing ABI",
                directAbiRejected && !File.Exists(badDirectWriterPath));
            byte[] unknownAbiBytes = File.ReadAllBytes(cubin10Path);
            unknownAbiBytes[7] = 0;
            unknownAbiBytes[8] = 0;
            string unknownAbiPath = Path.Combine(asmDir, "unknown-abi.cubin");
            File.WriteAllBytes(unknownAbiPath, unknownAbiBytes);
            Check("unknown cubin ABI is not guessed",
                CubinFile.Load(unknownAbiPath).ToolkitFromAbi == null);
            bool unknownAbiFatbinRejected = false;
            try
            {
                FatbinWriter.Write(
                    Path.Combine(asmDir, "unknown-abi.fatbin"),
                    unknownAbiBytes, 75);
            }
            catch (InvalidDataException)
            {
                unknownAbiFatbinRejected = true;
            }
            Check("fatbin rejects unknown ABI", unknownAbiFatbinRejected);
            Check("assembler IR output", File.Exists(irPath) &&
                File.ReadAllText(irPath).Contains("\"architecture\": \"sm_75\""));
            Check("assembler section", assembled.Sections.Count == 1 &&
                assembled.Sections[0].Data.Length == 32);
            Check("assembler regcount metadata",
                assembled.RegisterCounts.Count > 0 &&
                assembled.RegisterCounts.Values.Any(value => value == 2));
            string fatbinPath = Path.Combine(asmDir, "sample.fatbin");
            FatbinWriter.Write(
                fatbinPath, File.ReadAllBytes(cubinPath), assembled.Architecture);
            FatbinFile fatbin = FatbinFile.Parse(File.ReadAllBytes(fatbinPath));
            Check("assembler fatbin output",
                fatbin.Entries.Count == 1 &&
                CubinFile.Load(fatbin.Entries[0].Payload).ArchitectureKey == "sm_75");
            Check("fatbin writer CUDA 13.x code version",
                fatbin.Entries[0].CodeVersionMajor == 1 &&
                fatbin.Entries[0].CodeVersionMinor == 8);

            byte[] originalCubin = File.ReadAllBytes(cubinPath);
            var compressionCases = new[]
            {
                (FatbinCompression.None, "NONE"),
                (FatbinCompression.Deflate, "DEFLATE"),
                (FatbinCompression.Lz4, "LZ4"),
                (FatbinCompression.Lz4V2, "LZ4v2"),
                (FatbinCompression.Zstd, "ZSTD"),
            };
            foreach (var (compression, name) in compressionCases)
            {
                string compressedPath = Path.Combine(
                    asmDir, $"sample-{compression}.fatbin");
                FatbinWriter.Write(
                    compressedPath, originalCubin, assembled.Architecture, compression);
                FatbinEntry entry = FatbinFile.Parse(
                    File.ReadAllBytes(compressedPath)).Entries[0];
                Check($"fatbin writer {name} payload",
                    entry.Payload.AsSpan().SequenceEqual(originalCubin));
                Equal($"fatbin writer {name} flag", name, entry.Compression);
            }

            string multiFatbinPath = Path.Combine(asmDir, "sample-multi.fatbin");
            FatbinWriter.Write(
                multiFatbinPath,
                new[]
                {
                    new FatbinImage(originalCubin, assembled.Architecture),
                    new FatbinImage(
                        File.ReadAllBytes(cubin10Path), assembled10.Architecture),
                });
            byte[] multiFatbinBytes = File.ReadAllBytes(multiFatbinPath);
            FatbinFile multiFatbin = FatbinFile.Parse(multiFatbinBytes);
            Check("fatbin writer multi-entry count", multiFatbin.Entries.Count == 2);
            Check("fatbin writer multi-entry architectures",
                multiFatbin.Entries[0].Architecture == assembled.Architecture &&
                multiFatbin.Entries[1].Architecture == assembled10.Architecture);
            Check("fatbin writer multi-entry payloads",
                multiFatbin.Entries[0].Payload.AsSpan().SequenceEqual(originalCubin) &&
                multiFatbin.Entries[1].Payload.AsSpan().SequenceEqual(
                    File.ReadAllBytes(cubin10Path)));
            Check("fatbin writer cubin ptxas-options offset",
                BinaryPrimitives.ReadUInt32LittleEndian(
                    multiFatbinBytes.AsSpan(16 + 20)) == 0);

            string ptxInputPath = Path.Combine(asmDir, "sample.ptx");
            string ptxFatbinPath = Path.Combine(asmDir, "sample-ptx.fatbin");
            File.WriteAllText(ptxInputPath, """
                .version 6.0
                .target sm_86
                .address_size 64
                .visible .entry ptxk()
                {
                    ret;
                }
                """);
            FatbinWriter.Write(
                ptxFatbinPath,
                new[]
                {
                    new FatbinImage(
                        File.ReadAllBytes(ptxInputPath), 86, FatbinImageKind.Ptx),
                });
            byte[] ptxFatbinBytes = File.ReadAllBytes(ptxFatbinPath);
            FatbinEntry ptxEntry = FatbinFile.Parse(ptxFatbinBytes).Entries[0];
            Check("fatbin writer PTX kind", ptxEntry.IsPtx);
            Check("fatbin writer PTX code version",
                ptxEntry.CodeVersionMajor == 6 && ptxEntry.CodeVersionMinor == 0);
            Check("fatbin writer PTX payload",
                Encoding.UTF8.GetString(ptxEntry.Payload).Contains(
                    ".entry ptxk", StringComparison.Ordinal));
            Check("fatbin writer PTX header geometry",
                BinaryPrimitives.ReadUInt32LittleEndian(
                    ptxFatbinBytes.AsSpan(16 + 4)) == 80 &&
                BinaryPrimitives.ReadUInt32LittleEndian(
                    ptxFatbinBytes.AsSpan(16 + 20)) == 64);

            string ptx102InputPath = Path.Combine(asmDir, "sample-102.ptx");
            string ptx102FatbinPath = Path.Combine(asmDir, "sample-102-ptx.fatbin");
            File.WriteAllText(ptx102InputPath, """
                .version 6.0
                .target sm_75
                .address_size 64
                .visible .entry ptxk()
                {
                    ret;
                }
                """);
            FatbinWriter.Write(
                ptx102FatbinPath,
                new[]
                {
                    new FatbinImage(
                        File.ReadAllBytes(ptx102InputPath), 75, FatbinImageKind.Ptx),
                },
                toolkit: FatbinToolkit.Cuda102);
            byte[] ptx102FatbinBytes = File.ReadAllBytes(ptx102FatbinPath);
            FatbinEntry ptx102Entry = FatbinFile.Parse(ptx102FatbinBytes).Entries[0];
            Check("fatbin writer CUDA 10.2 PTX header geometry",
                BinaryPrimitives.ReadUInt32LittleEndian(
                    ptx102FatbinBytes.AsSpan(16 + 4)) == 72 &&
                BinaryPrimitives.ReadUInt32LittleEndian(
                    ptx102FatbinBytes.AsSpan(16 + 20)) == 64 &&
                ptx102Entry.CodeVersionMajor == 6 &&
                ptx102Entry.CodeVersionMinor == 0);

            string sm100SassPath = Path.Combine(asmDir, "sample-100.sass");
            string sm100CubinPath = Path.Combine(asmDir, "sample-100.cubin");
            File.WriteAllText(sm100SassPath, """
                .target sm_100
                .section .text.sm100,"ax",@progbits
                .global sm100
                sm100:
                    MOV R0, 0x1;
                    EXIT;
                """);
            SassAssembler.Assemble(
                sm100SassPath, "sm_100", sm100CubinPath, null, "13.4");
            string sm100FatbinPath = Path.Combine(asmDir, "sample-100.fatbin");
            FatbinWriter.Write(
                sm100FatbinPath,
                new[]
                {
                    new FatbinImage(
                        File.ReadAllBytes(sm100CubinPath), 100,
                        ArchitectureKey: "sm_100"),
                });
            byte[] sm100FatbinBytes = File.ReadAllBytes(sm100FatbinPath);
            // FatbinWriter defaults to the host platform, so the flags are 0x41 on
            // Windows and 0x11 on Linux.  This check used to hard-code the Windows
            // value and fail on every non-Windows test run.
            ulong hostPlatformFlags = OperatingSystem.IsWindows() ? 0x41UL : 0x11UL;
            Check("fatbin writer SM100 extended header",
                BinaryPrimitives.ReadUInt32LittleEndian(
                    sm100FatbinBytes.AsSpan(16 + 4)) == 120 &&
                BinaryPrimitives.ReadUInt32LittleEndian(
                    sm100FatbinBytes.AsSpan(16 + 20)) == 0x40 &&
                BinaryPrimitives.ReadUInt64LittleEndian(
                    sm100FatbinBytes.AsSpan(16 + 40)) ==
                    (0x1000000UL | hostPlatformFlags) &&
                BinaryPrimitives.ReadUInt32LittleEndian(
                    sm100FatbinBytes.AsSpan(16 + 64)) == 0x48);

            string identifiedFatbinPath = Path.Combine(asmDir, "identified.fatbin");
            FatbinWriter.Write(
                identifiedFatbinPath,
                new[]
                {
                    new FatbinImage(
                        originalCubin, assembled.Architecture,
                        FatbinImageKind.Elf, assembled.ArchitectureKey, "testid"),
                });
            FatbinEntry identifiedEntry = FatbinFile.Parse(
                File.ReadAllBytes(identifiedFatbinPath)).Entries[0];
            Check("fatbin writer identifier metadata",
                identifiedEntry.Identifier == "testid" &&
                BinaryPrimitives.ReadUInt32LittleEndian(
                    File.ReadAllBytes(identifiedFatbinPath).AsSpan(16 + 4)) == 72);

            string optionsFatbinPath = Path.Combine(asmDir, "ptx-options.fatbin");
            FatbinWriter.Write(
                optionsFatbinPath,
                new[]
                {
                    new FatbinImage(
                        File.ReadAllBytes(ptxInputPath), 86, FatbinImageKind.Ptx,
                        "sm_86", null, "-O3"),
                });
            FatbinEntry ptxOptionsEntry = FatbinFile.Parse(
                File.ReadAllBytes(optionsFatbinPath)).Entries[0];
            Check("fatbin writer PTX options metadata",
                ptxOptionsEntry.PtxasOptions == "-O3");

            byte[] ltoPayload =
            {
                0x42, 0x43, 0xC0, 0xDE, 0x35, 0x14, 0, 0,
                5, 0, 0, 0, 0, 0, 0, 0,
            };
            string ltoFatbinPath = Path.Combine(asmDir, "ltoir.fatbin");
            FatbinWriter.Write(
                ltoFatbinPath,
                new[]
                {
                    new FatbinImage(
                        ltoPayload, 86, FatbinImageKind.Ltoir, "sm_86"),
                });
            FatbinEntry ltoEntry = FatbinFile.Parse(
                File.ReadAllBytes(ltoFatbinPath)).Entries[0];
            Check("fatbin writer LTOIR entry",
                ltoEntry.IsLtoIr &&
                ltoEntry.CodeVersionMajor == 53 &&
                ltoEntry.CodeVersionMinor == 20 &&
                ltoEntry.Payload.AsSpan().SequenceEqual(ltoPayload));

            byte[] tilePayload = { 1, 2, 3, 4, 5, 6, 7, 8 };
            string tileFatbinPath = Path.Combine(asmDir, "tileir.fatbin");
            FatbinWriter.Write(
                tileFatbinPath,
                new[]
                {
                    new FatbinImage(
                        tilePayload, 100, FatbinImageKind.TileIr, "sm_100"),
                });
            FatbinEntry tileWriterEntry = FatbinFile.Parse(
                File.ReadAllBytes(tileFatbinPath)).Entries[0];
            Check("fatbin writer TileIR entry",
                tileWriterEntry.IsTileIr &&
                tileWriterEntry.Architecture == 100 &&
                tileWriterEntry.Payload.AsSpan().SequenceEqual(tilePayload));

            byte[] indexPayload = { 9, 8, 7, 6 };
            string indexFatbinPath = Path.Combine(asmDir, "index.fatbin");
            FatbinWriter.Write(
                indexFatbinPath,
                new[]
                {
                    new FatbinImage(indexPayload, 0, FatbinImageKind.Index),
                });
            FatbinEntry indexEntry = FatbinFile.Parse(
                File.ReadAllBytes(indexFatbinPath)).Entries[0];
            Check("fatbin writer index entry",
                indexEntry.Kind == 0x20 &&
                indexEntry.Payload.Length >= indexPayload.Length &&
                indexEntry.Payload.AsSpan(0, indexPayload.Length)
                    .SequenceEqual(indexPayload));

            bool cuda102ModernRejected = false;
            try
            {
                FatbinWriter.Write(
                    Path.Combine(asmDir, "cuda102-modern.fatbin"),
                    new[] { new FatbinImage(originalCubin, assembled.Architecture) },
                    toolkit: FatbinToolkit.Cuda102);
            }
            catch (InvalidDataException)
            {
                cuda102ModernRejected = true;
            }
            Check("fatbin writer CUDA 10.2 rejects modern ABI",
                cuda102ModernRejected);

            string cuda102FatbinPath = Path.Combine(asmDir, "sample-102.fatbin");
            FatbinWriter.Write(
                cuda102FatbinPath, File.ReadAllBytes(cubin10Path),
                assembled10.Architecture, FatbinCompression.Lz4);
            FatbinEntry writer102Entry = FatbinFile.Parse(
                File.ReadAllBytes(cuda102FatbinPath)).Entries[0];
            Check("fatbin writer CUDA 10.x code version",
                writer102Entry.CodeVersionMajor == 1 &&
                writer102Entry.CodeVersionMinor == 7);

            InstructionDecoder.UseArchitecture(75, "13.4");
            ulong low = BitConverter.ToUInt64(assembled.Sections[0].Data, 0);
            ulong high = BitConverter.ToUInt64(assembled.Sections[0].Data, 8);
            Equal("assembler mov roundtrip",
                "MOV R1, c[0x0][0x28]",
                InstructionDecoder.Decode(low, high, 0).Text);

            string sm70Path = Path.Combine(asmDir, "sample70.sass");
            string sm70CubinPath = Path.Combine(asmDir, "sample70.cubin");
            File.WriteAllText(sm70Path, """
                .target sm_70
                .section .text.test70,"ax",@progbits
                .global test70
                test70:
                    MOV R1, c[0x0][0x28];
                    NOP;
                    EXIT;
                """);
            SassAssembler.Assemble(sm70Path, "sm_70", sm70CubinPath);
            byte[] sm70Image = File.ReadAllBytes(sm70CubinPath);
            Check("assembler sm70 legacy elf flags",
                BinaryPrimitives.ReadUInt32LittleEndian(sm70Image.AsSpan(0x30)) ==
                    0x460546 &&
                sm70Image[7] == 0x33 && sm70Image[8] == 7);

            string sm86Path = Path.Combine(asmDir, "sample86.sass");
            string sm86CubinPath = Path.Combine(asmDir, "sample86.cubin");
            File.WriteAllText(sm86Path, """
                .target sm_86
                .section .text.test86,"ax",@progbits
                .global test86
                test86:
                    LDC.64 R2, c[0x0][R2+0x160];
                    LOP3.LUT R2, R0, 0x3, RZ, 0xc0, !PT;
                    STG.E [R2.64], R0;
                    EXIT;
                """);
            SassAssembler.Assemble(sm86Path, "sm_86", sm86CubinPath);
            CubinFile assembled86 = CubinFile.Load(sm86CubinPath);
            Check("assembler sm86 section", assembled86.Sections.Count == 1 &&
                assembled86.Sections[0].Data.Length == 64);
            InstructionDecoder.UseArchitecture(86);
            ulong low86 = BitConverter.ToUInt64(assembled86.Sections[0].Data, 0);
            ulong high86 = BitConverter.ToUInt64(assembled86.Sections[0].Data, 8);
            Equal("assembler ldc roundtrip",
                "LDC.64 R2, c[0x0][R2+0x160]",
                InstructionDecoder.Decode(low86, high86, 0).Text);

            string extraPath = Path.Combine(asmDir, "extra.sass");
            string extraCubinPath = Path.Combine(asmDir, "extra.cubin");
            File.WriteAllText(extraPath, """
                .target sm_86
                .section .text.extra,"ax",@progbits
                .sectioninfo @"SHI_REGISTERS=16"
                .global extra
                extra:
                    BSSY B0, `(.L_extra);
                    NOP;
                    BSYNC B0;
                .L_extra:
                    IADD3 R5, R0, -c[0x0][0x168], RZ;
                    FFMA.RZ R0, R13.reuse, R11.reuse, R10.reuse;
                    EXIT;
                    RET.REL.NODEC R6 `(extra);
                    BRX R6 (*"BRANCH_TARGETS .L_extra"*);
                """);
            SassAssembler.Assemble(extraPath, "sm_86", extraCubinPath);
            CubinFile assembledExtra = CubinFile.Load(extraCubinPath);
            Check("assembler branch/marker section",
                assembledExtra.Sections.Count == 1 &&
                assembledExtra.Sections[0].Data.Length == 128);
            InstructionDecoder.UseArchitecture(86);
            ulong extraLow = BitConverter.ToUInt64(
                assembledExtra.Sections[0].Data, 0);
            ulong extraHigh = BitConverter.ToUInt64(
                assembledExtra.Sections[0].Data, 8);
            Check("assembler bssy class",
                InstructionDecoder.Decode(extraLow, extraHigh, 0).ClassName ==
                "bssy_");
            ulong ffmaLow = BitConverter.ToUInt64(
                assembledExtra.Sections[0].Data, 0x40);
            ulong ffmaHigh = BitConverter.ToUInt64(
                assembledExtra.Sections[0].Data, 0x48);
            Equal("assembler reuse roundtrip",
                "FFMA.RZ R0, R13.reuse, R11.reuse, R10.reuse",
                InstructionDecoder.Decode(ffmaLow, ffmaHigh, 0x40).Text);
            Check("assembler sectioninfo register count",
                assembledExtra.RegisterCounts.Values.Any(value => value == 16));
            Check("assembler brx metadata",
                assembledExtra.Sections[0].IndirectBranchTargets.TryGetValue(
                    0x70, out List<ulong>? extraTargets) &&
                extraTargets.Count == 1 &&
                extraTargets[0] == 0x30);

            string widePath = Path.Combine(asmDir, "wide.sass");
            string wideCubinPath = Path.Combine(asmDir, "wide.cubin");
            File.WriteAllText(widePath, """
                .target sm_86
                .section .text.wide,"ax",@progbits
                .global wide
                wide:
                    @!UP0 UIMAD UR5, UR10, UR4, URZ;
                    LDG.E.U16.CONSTANT R28, desc[UR4][R28.64];
                    HSETP2.NEU.AND P0, PT, RZ.H0_H0, c[0x0] [0x178].H1_H1, PT;
                    TEX.SCR.LL RZ, R0, R36, R30, 0x0, 0x5c, CUBE, 0x1;
                    IDP.4A.S8.U8 R55, R55, c[0x2][0x4], RZ;
                    LDGSTS.E.LTC128B.128.CONSTANT [R84+UR6], desc[UR7][R82.64], P3;
                    EXIT;
                """);
            SassAssembler.Assemble(widePath, "sm_86", wideCubinPath);
            CubinFile assembledWide = CubinFile.Load(wideCubinPath);
            Check("assembler descriptor/marker forms",
                assembledWide.Sections.Count == 1 &&
                assembledWide.Sections[0].Data.Length == 112);
            ulong texLow = BitConverter.ToUInt64(
                assembledWide.Sections[0].Data, 0x30);
            ulong texHigh = BitConverter.ToUInt64(
                assembledWide.Sections[0].Data, 0x38);
            Equal("assembler texture roundtrip",
                "TEX.SCR.LL RZ, R0, R36, R30, 0x0, 0x5c, CUBE, 0x1",
                InstructionDecoder.Decode(texLow, texHigh, 0x30).Text);

            string relocPath = Path.Combine(asmDir, "reloc.sass");
            string relocCubinPath = Path.Combine(asmDir, "reloc.cubin");
            File.WriteAllText(relocPath, """
                .target sm_86
                .section .text.reloc,"ax",@progbits
                .global reloc
                reloc:
                    MOV R20, 32@lo((reloc + .L_reloc@srel));
                .L_reloc:
                    EXIT;
                """);
            SassAssembler.Assemble(relocPath, "sm_86", relocCubinPath);
            CubinFile assembledReloc = CubinFile.Load(relocCubinPath);
            Check("assembler relocation section",
                assembledReloc.Sections.Count == 1 &&
                assembledReloc.Sections[0].Relocations.Count == 1 &&
                assembledReloc.Sections[0].Relocations[0].Type == 0x38 &&
                assembledReloc.Sections[0].Relocations[0].Offset == 0);

            string urzPath = Path.Combine(asmDir, "urz.sass");
            string urzCubinPath = Path.Combine(asmDir, "urz.cubin");
            File.WriteAllText(urzPath, """
                .target sm_110
                .section .text.urz,"ax",@progbits
                .global urz
                urz:
                    UPRMT UR4, UR4, 0x8880, URZ;
                    EXIT;
                """);
            SassAssembler.Assemble(urzPath, "sm_110", urzCubinPath);
            CubinFile assembledUrz = CubinFile.Load(urzCubinPath);
            InstructionDecoder.UseArchitecture(110);
            ulong urzLow = BitConverter.ToUInt64(
                assembledUrz.Sections[0].Data, 0);
            ulong urzHigh = BitConverter.ToUInt64(
                assembledUrz.Sections[0].Data, 8);
            Equal("assembler architecture-specific URZ",
                "UPRMT UR4, UR4, 0x8880, URZ",
                InstructionDecoder.Decode(urzLow, urzHigh, 0).Text);
            InstructionDecoder.UseArchitecture(86);

            string callRelPath = Path.Combine(asmDir, "callrel.sass");
            string callRelCubinPath = Path.Combine(asmDir, "callrel.cubin");
            File.WriteAllText(callRelPath, """
                .target sm_100
                .section .text.callrel,"ax",@progbits
                .global callrel
                callrel:
                    CALL.ABS.NOINC `(lookup);
                    EXIT;
                lookup:
                    EXIT;
                """);
            SassAssembler.Assemble(callRelPath, "sm_100", callRelCubinPath);
            CubinFile assembledCallRel = CubinFile.Load(callRelCubinPath);
            Check("assembler absolute-call relocation",
                assembledCallRel.Sections.Count == 1 &&
                assembledCallRel.Sections[0].Relocations.Count == 1 &&
                assembledCallRel.Sections[0].Relocations[0].Type == 0x4b);

            string callRel86Path = Path.Combine(asmDir, "callrel86.sass");
            string callRel86CubinPath = Path.Combine(asmDir, "callrel86.cubin");
            File.WriteAllText(callRel86Path, """
                .target sm_86
                .section .text.callrel,"ax",@progbits
                .global callrel
                callrel:
                    CALL.ABS.NOINC `(lookup);
                    EXIT;
                lookup:
                    EXIT;
                """);
            SassAssembler.Assemble(callRel86Path, "sm_86", callRel86CubinPath);
            CubinFile assembledCallRel86 = CubinFile.Load(callRel86CubinPath);
            Check("assembler pre-SM90 absolute call has no relocation",
                assembledCallRel86.Sections.Count == 1 &&
                assembledCallRel86.Sections[0].Relocations.Count == 0);
            InstructionDecoder.UseArchitecture(86);
            ulong call86Low = BitConverter.ToUInt64(
                assembledCallRel86.Sections[0].Data, 0);
            ulong call86High = BitConverter.ToUInt64(
                assembledCallRel86.Sections[0].Data, 8);
            Equal("assembler pre-SM90 absolute call target",
                "CALL.ABS.NOINC 0x20",
                InstructionDecoder.Decode(call86Low, call86High, 0).Text);

            string callRel70Path = Path.Combine(asmDir, "callrel70.sass");
            string callRel70CubinPath = Path.Combine(asmDir, "callrel70.cubin");
            File.WriteAllText(callRel70Path, """
                .target sm_70
                .section .text.callrel70,"ax",@progbits
                .global callrel70
                callrel70:
                    CALL.REL.NOINC `(lookup);
                    EXIT;
                lookup:
                    EXIT;
                """);
            SassAssembler.Assemble(callRel70Path, "sm_70", callRel70CubinPath);
            CubinFile assembledCallRel70 = CubinFile.Load(callRel70CubinPath);
            InstructionDecoder.UseArchitecture(70);
            ulong call70Low = BitConverter.ToUInt64(
                assembledCallRel70.Sections[0].Data, 0);
            ulong call70High = BitConverter.ToUInt64(
                assembledCallRel70.Sections[0].Data, 8);
            Check("assembler relative call keeps REL class",
                InstructionDecoder.Decode(call70Low, call70High, 0).Text
                    .StartsWith("CALL.REL.NOINC", StringComparison.Ordinal));
            InstructionDecoder.UseArchitecture(86);

            string retAbsPath = Path.Combine(asmDir, "retabs.sass");
            string retAbsCubinPath = Path.Combine(asmDir, "retabs.cubin");
            File.WriteAllText(retAbsPath, """
                .target sm_75
                .section .text.retabs,"ax",@progbits
                .global retabs
                retabs:
                    RET.ABS.NODEC R20 -0x420;
                """);
            SassAssembler.Assemble(
                retAbsPath, "sm_75", retAbsCubinPath, null, "13.4");
            CubinFile assembledRetAbs = CubinFile.Load(retAbsCubinPath);
            InstructionDecoder.UseArchitecture(75, "13.4");
            ulong retAbsLow = BitConverter.ToUInt64(
                assembledRetAbs.Sections[0].Data, 0);
            ulong retAbsHigh = BitConverter.ToUInt64(
                assembledRetAbs.Sections[0].Data, 8);
            Equal("assembler negative absolute ret target",
                "RET.ABS.NODEC R20 -0x420",
                InstructionDecoder.Decode(retAbsLow, retAbsHigh, 0).Text);
            InstructionDecoder.UseArchitecture(86);

            string registerCallPath = Path.Combine(asmDir, "regcall.sass");
            string registerCallCubinPath = Path.Combine(asmDir, "regcall.cubin");
            File.WriteAllText(registerCallPath, """
                .target sm_120
                .section .text.regcall,"ax",@progbits
                .global regcall
                regcall:
                    CALL.ABS.NOINC R2 (*"RELOC 114 __UFT_OFFSET 0"*);
                    EXIT;
                """);
            SassAssembler.Assemble(registerCallPath, "sm_120", registerCallCubinPath);
            CubinFile assembledRegisterCall = CubinFile.Load(registerCallCubinPath);
            Check("assembler preserves register-indirect call relocation",
                assembledRegisterCall.Sections.Count == 1 &&
                assembledRegisterCall.Sections[0].Relocations.Count == 1 &&
                assembledRegisterCall.Sections[0].Relocations[0].Type == 0x72 &&
                assembledRegisterCall.Sections[0].Relocations[0].SymbolName ==
                    "__UFT_OFFSET");

            string umovRelPath = Path.Combine(asmDir, "umovrel.sass");
            string umovRelCubinPath = Path.Combine(asmDir, "umovrel.cubin");
            File.WriteAllText(umovRelPath, """
                .target sm_100
                .section .text.umovrel,"ax",@progbits
                .global umovrel
                umovrel:
                    UMOV UR4, `(lookup);
                    EXIT;
                lookup:
                    EXIT;
                """);
            SassAssembler.Assemble(umovRelPath, "sm_100", umovRelCubinPath);
            CubinFile assembledUmovRel = CubinFile.Load(umovRelCubinPath);
            Check("assembler UMOV relocation",
                assembledUmovRel.Sections.Count == 1 &&
                assembledUmovRel.Sections[0].Relocations.Count == 1 &&
                assembledUmovRel.Sections[0].Relocations[0].Type == 0x37);

            string ldcRelPath = Path.Combine(asmDir, "ldcrel.sass");
            string ldcRelCubinPath = Path.Combine(asmDir, "ldcrel.cubin");
            File.WriteAllText(ldcRelPath, """
                .target sm_86
                .section .text.ldcrel,"ax",@progbits
                .global ldcrel
                ldcrel:
                    LDC R6, c[R8+`((lookup + -0x8000))];
                    EXIT;
                lookup:
                    EXIT;
                """);
            SassAssembler.Assemble(ldcRelPath, "sm_86", ldcRelCubinPath);
            CubinFile assembledLdcRel = CubinFile.Load(ldcRelCubinPath);
            Check("assembler symbolic constant relocation",
                assembledLdcRel.Sections.Count == 1 &&
                assembledLdcRel.Sections[0].Relocations.Count == 1 &&
                assembledLdcRel.Sections[0].Relocations[0].Type == 0x42 &&
                assembledLdcRel.Sections[0].Relocations[0].Addend == -0x8000);

            string hiddenRelPath = Path.Combine(asmDir, "hiddenrel.sass");
            string hiddenRelCubinPath = Path.Combine(asmDir, "hiddenrel.cubin");
            File.WriteAllText(hiddenRelPath, """
                .target sm_90
                .section .text.hiddenrel,"ax",@progbits
                .global hiddenrel
                hiddenrel:
                    LDC R6, c[0x2][R8] (*"RELOC 66 missing_symbol -0x8000"*);
                    EXIT;
                """);
            SassAssembler.Assemble(hiddenRelPath, "sm_90", hiddenRelCubinPath);
            CubinFile assembledHiddenRel = CubinFile.Load(hiddenRelCubinPath);
            Check("assembler preserved hidden relocation",
                assembledHiddenRel.Sections.Count == 1 &&
                assembledHiddenRel.Sections[0].Relocations.Count == 1 &&
                assembledHiddenRel.Sections[0].Relocations[0].Type == 0x42 &&
                assembledHiddenRel.Sections[0].Relocations[0].Addend == -0x8000 &&
                assembledHiddenRel.Sections[0].Relocations[0].SymbolName ==
                    "missing_symbol");

            string relRelPath = Path.Combine(asmDir, "relrel.sass");
            string relRelCubinPath = Path.Combine(asmDir, "relrel.cubin");
            File.WriteAllText(relRelPath, """
                .target sm_90
                .section .text.relrel,"ax",@progbits
                .global relrel
                relrel:
                    UMOV UR4, 0x0 (*"RELOCREL 55 missing_symbol 0"*);
                    EXIT;
                """);
            SassAssembler.Assemble(relRelPath, "sm_90", relRelCubinPath);
            CubinFile assembledRelRel = CubinFile.Load(relRelCubinPath);
            Check("assembler preserved SHT_REL relocation",
                assembledRelRel.Sections.Count == 1 &&
                assembledRelRel.Sections[0].Relocations.Count == 1 &&
                assembledRelRel.Sections[0].Relocations[0].Type == 0x37 &&
                assembledRelRel.Sections[0].Relocations[0].IsRel &&
                assembledRelRel.Sections[0].Relocations[0].SymbolName ==
                    "missing_symbol");

            string sharedPath = Path.Combine(asmDir, "sharedrel.sass");
            string sharedCubinPath = Path.Combine(asmDir, "sharedrel.cubin");
            File.WriteAllText(sharedPath, """
                .target sm_86
                .section .text.sharedrel,"ax",@progbits
                .global sharedrel
                sharedrel:
                    STS [R5+`(shrdMem32)], R4;
                    EXIT;
                """);
            SassAssembler.Assemble(sharedPath, "sm_86", sharedCubinPath);
            CubinFile assembledShared = CubinFile.Load(sharedCubinPath);
            Check("assembler shared-memory symbol relocation",
                assembledShared.Sections.Count == 1 &&
                assembledShared.Sections[0].Relocations.Count == 1 &&
                assembledShared.Sections[0].Relocations[0].Type == 0x4a &&
                assembledShared.Sections[0].Relocations[0].IsRel &&
                assembledShared.Sections[0].Relocations[0].SymbolName ==
                    "shrdMem32");

            string constTblPath = Path.Combine(asmDir, "consttbl.sass");
            string constTblCubinPath = Path.Combine(asmDir, "consttbl.cubin");
            File.WriteAllText(constTblPath, """
                .target sm_89
                .section .text.consttbl,"ax",@progbits
                .global consttbl
                consttbl:
                    MOV R0, `(constTblFullTrilinearLevels);
                    EXIT;
                """);
            SassAssembler.Assemble(constTblPath, "sm_89", constTblCubinPath);
            CubinFile assembledConstTbl = CubinFile.Load(constTblCubinPath);
            Check("assembler constant-table symbol relocation",
                assembledConstTbl.Sections.Count == 1 &&
                assembledConstTbl.Sections[0].Relocations.Count == 1 &&
                assembledConstTbl.Sections[0].Relocations[0].Type == 0x3b &&
                assembledConstTbl.Sections[0].Relocations[0].IsRel &&
                assembledConstTbl.Sections[0].Relocations[0].SymbolName ==
                    "constTblFullTrilinearLevels");

            string ldsPath = Path.Combine(asmDir, "lds64.sass");
            string ldsCubinPath = Path.Combine(asmDir, "lds64.cubin");
            File.WriteAllText(ldsPath, """
                .target sm_75
                .section .text.lds64,"ax",@progbits
                .global lds64
                lds64:
                    LDS.U.64 R4, [R0.X8];
                    EXIT;
                """);
            SassAssembler.Assemble(ldsPath, "sm_75", ldsCubinPath, null, "13.4");
            CubinFile assembledLds = CubinFile.Load(ldsCubinPath);
            InstructionDecoder.UseArchitecture(75, "13.4");
            ulong ldsLow = BitConverter.ToUInt64(assembledLds.Sections[0].Data, 0);
            ulong ldsHigh = BitConverter.ToUInt64(assembledLds.Sections[0].Data, 8);
            Equal("assembler LDS.U.64",
                "LDS.U.64R4,[R0.X8]",
                InstructionDecoder.Decode(ldsLow, ldsHigh, 0).Text
                    .Replace(" ", "", StringComparison.Ordinal));

            string depbarPath = Path.Combine(asmDir, "depbar.sass");
            string depbarCubinPath = Path.Combine(asmDir, "depbar.cubin");
            File.WriteAllText(depbarPath, """
                .target sm_80
                .section .text.depbar,"ax",@progbits
                .global depbar
                depbar:
                    DEPBAR.LE SB1, 0x0, {3,2};
                    EXIT;
                """);
            SassAssembler.Assemble(depbarPath, "sm_80", depbarCubinPath);
            CubinFile assembledDepbar = CubinFile.Load(depbarCubinPath);
            InstructionDecoder.UseArchitecture(80);
            ulong depbarLow = BitConverter.ToUInt64(
                assembledDepbar.Sections[0].Data, 0);
            ulong depbarHigh = BitConverter.ToUInt64(
                assembledDepbar.Sections[0].Data, 8);
            Equal("assembler DEPBAR bitset",
                "DEPBAR.LESB1,0x0,{3,2}",
                InstructionDecoder.Decode(depbarLow, depbarHigh, 0).Text
                    .Replace(" ", "", StringComparison.Ordinal));

            string fselPath = Path.Combine(asmDir, "fsel.sass");
            string fselCubinPath = Path.Combine(asmDir, "fsel.cubin");
            File.WriteAllText(fselPath, """
                .target sm_89
                .section .text.fsel,"ax",@progbits
                .global fsel
                fsel:
                    FSEL R4, RZ, +QNAN, !P1;
                    EXIT;
                """);
            SassAssembler.Assemble(fselPath, "sm_89", fselCubinPath);
            CubinFile assembledFsel = CubinFile.Load(fselCubinPath);
            InstructionDecoder.UseArchitecture(89);
            ulong fselLow = BitConverter.ToUInt64(assembledFsel.Sections[0].Data, 0);
            ulong fselHigh = BitConverter.ToUInt64(assembledFsel.Sections[0].Data, 8);
            Equal("assembler QNAN immediate",
                "FSELR4,RZ,+QNAN,!P1",
                InstructionDecoder.Decode(fselLow, fselHigh, 0).Text
                    .Replace(" ", "", StringComparison.Ordinal));

            string brxNumericPath = Path.Combine(asmDir, "brxnumeric.sass");
            string brxNumericCubinPath = Path.Combine(asmDir, "brxnumeric.cubin");
            File.WriteAllText(brxNumericPath, """
                .target sm_86
                .section .text.brxnumeric,"ax",@progbits
                .global brxnumeric
                brxnumeric:
                    BRX R6 -0x4c0;
                    EXIT;
                """);
            SassAssembler.Assemble(brxNumericPath, "sm_86", brxNumericCubinPath);
            CubinFile assembledBrxNumeric = CubinFile.Load(brxNumericCubinPath);
            InstructionDecoder.UseArchitecture(86);
            ulong brxLow = BitConverter.ToUInt64(
                assembledBrxNumeric.Sections[0].Data, 0);
            ulong brxHigh = BitConverter.ToUInt64(
                assembledBrxNumeric.Sections[0].Data, 8);
            Check("assembler BRX numeric annotation",
                InstructionDecoder.Decode(brxLow, brxHigh, 0).Text
                    .StartsWith("BRX R6", StringComparison.Ordinal));

            string cxPath = Path.Combine(asmDir, "cx.sass");
            string cxCubinPath = Path.Combine(asmDir, "cx.cubin");
            File.WriteAllText(cxPath, """
                .target sm_86
                .section .text.cx,"ax",@progbits
                .global cx
                cx:
                    IMAD R2, R2, cx[UR0][0x0], R3;
                    EXIT;
                """);
            SassAssembler.Assemble(cxPath, "sm_86", cxCubinPath);
            CubinFile assembledCx = CubinFile.Load(cxCubinPath);
            InstructionDecoder.UseArchitecture(86);
            ulong cxLow = BitConverter.ToUInt64(assembledCx.Sections[0].Data, 0);
            ulong cxHigh = BitConverter.ToUInt64(assembledCx.Sections[0].Data, 8);
            Equal("assembler CX constant",
                "IMAD R2, R2, cx[UR0][0x0], R3",
                InstructionDecoder.Decode(cxLow, cxHigh, 0).Text);

            string warningPath = Path.Combine(asmDir, "sm101-warning.sass");
            string warningCubinPath = Path.Combine(asmDir, "sm101-warning.cubin");
            File.WriteAllText(warningPath, """
                .target sm_101
                .section .text.sm101,"ax",@progbits
                .global sm101
                sm101:
                    MOV R1, 0x28;
                    EXIT;
                """);
            TextWriter originalError = Console.Error;
            var warningOutput = new StringWriter();
            try
            {
                Console.SetError(warningOutput);
                SassAssembler.Assemble(warningPath, "sm_101", warningCubinPath);
            }
            finally
            {
                Console.SetError(originalError);
            }
            Check("SM101 assembly warning",
                warningOutput.ToString().Contains(
                    "SM101 was renamed to SM110 in PTX ISA 9.0",
                    StringComparison.Ordinal));
            Check("SM101 assembles as SM110",
                CubinFile.Load(warningCubinPath).ArchitectureKey == "sm_110");

            foreach (string arch in new[]
            {
                "sm_75", "sm_80", "sm_86", "sm_87", "sm_88", "sm_89",
                "sm_90", "sm_90a", "sm_100", "sm_101", "sm_103", "sm_107",
                "sm_110", "sm_120", "sm_121",
                "sm_100a", "sm_100f", "sm_101a", "sm_101f", "sm_103a",
                "sm_103f", "sm_107a", "sm_107f", "sm_110a", "sm_110f",
                "sm_120a", "sm_120f", "sm_121a", "sm_121f",
            })
            {
                string archPath = Path.Combine(asmDir, arch + ".sass");
                string archCubinPath = Path.Combine(asmDir, arch + ".cubin");
                File.WriteAllText(archPath, $"""
                    .target {arch}
                    .section .text.{arch},"ax",@progbits
                    .global {arch}
                    {arch}:
                        MOV R1, 0x28;
                        EXIT;
                    """);
                SassAssembler.Assemble(
                    archPath, arch, archCubinPath, null,
                    arch == "sm_75" ? "13.4" : null);
                CubinFile archCubin = CubinFile.Load(archCubinPath);
                string expectedKey = arch switch
                {
                    "sm_101" => "sm_110",
                    "sm_101a" => "sm_110a",
                    "sm_101f" => "sm_110f",
                    _ => arch,
                };
                Check($"assembler {arch} compatibility",
                    archCubin.ArchitectureKey == expectedKey &&
                    archCubin.Sections.Count == 1 &&
                    archCubin.Sections[0].Data.Length == 32,
                    $"key = {archCubin.ArchitectureKey}");
            }

            string missingAbiPath = Path.Combine(asmDir, "missing-abi.sass");
            File.WriteAllText(missingAbiPath, """
                .target sm_80
                .section .text.missing,"ax",@progbits
                missing:
                    MOV R1, 0x28;
                    EXIT;
                """);
            string missingAbiCubin = Path.Combine(asmDir, "missing-abi.cubin");
            string missingAbiIr = Path.Combine(asmDir, "missing-abi.ir.json");
            bool missingAbiRejected = false;
            try
            {
                SassAssembler.Assemble(
                    missingAbiPath, "sm_80", missingAbiCubin,
                    missingAbiIr, "10.2");
            }
            catch (NotSupportedException)
            {
                missingAbiRejected = true;
            }
            Check("assembler rejects missing ABI", missingAbiRejected);
            Check("missing ABI writes no artifacts",
                !File.Exists(missingAbiCubin) && !File.Exists(missingAbiIr));

            File.WriteAllText(missingAbiPath, """
                .target sm_35
                .section .text.missing,"ax",@progbits
                missing:
                    MOV R1, 0x28;
                    EXIT;
                """);
            bool preVoltaAbiRejected = false;
            try
            {
                SassAssembler.Assemble(
                    missingAbiPath, "sm_35",
                    Path.Combine(asmDir, "missing-abi-35.cubin"), null, "13.4");
            }
            catch (NotSupportedException)
            {
                preVoltaAbiRejected = true;
            }
            Check("assembler rejects pre-Volta ABI", preVoltaAbiRejected);

            InstructionDecoder.UseArchitecture(86);
        }
        finally
        {
            try { Directory.Delete(asmDir, true); } catch { }
        }

        string apiDir = Path.Combine(
            Path.GetTempPath(), "rtxdisam-api-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(apiDir);
        try
        {
            string apiSassPath = Path.Combine(apiDir, "api.sass");
            string apiCubinPath = Path.Combine(apiDir, "api.cubin");
            File.WriteAllText(apiSassPath, """
                .target sm_50
                .section .text.api,"ax",@progbits
                .global api
                api:
                    MOV R0, c[0x0][0x140];
                    MOV32I R1, 0x148;
                    LDG.E R2, [R0];
                    STG.E [R1], R2;
                    EXIT;
                """);
            SassAssembler.Assemble(apiSassPath, "sm_50", apiCubinPath, null, "10.2");
            byte[] apiBytes = File.ReadAllBytes(apiCubinPath);

            RtxDisamText sassOnly = RtxDisamApi.ContainerToText(
                apiBytes, output: RtxDisamOutput.Sass);
            Check("managed API SASS-only output",
                sassOnly.Sass != null && sassOnly.Ptx == null &&
                sassOnly.Sass.Contains(".text.api", StringComparison.Ordinal),
                "SASS section was not emitted");

            var apiParameters = new[]
            {
                new PtxParameter("out", "u64", 0x140, 8),
            };
            RtxDisamText ptxOnly = RtxDisamApi.ContainerToText(
                apiBytes, "api", apiParameters, RtxDisamOutput.Ptx);
            Check("managed API PTX-only output",
                ptxOnly.Sass == null && ptxOnly.Ptx != null &&
                ptxOnly.Ptx.Contains(".entry api", StringComparison.Ordinal) &&
                ptxOnly.Ptx.Contains(".param .u64 out", StringComparison.Ordinal),
                ptxOnly.Ptx ?? "no PTX was emitted");

            RtxDisamText ptxDefault = RtxDisamApi.ContainerToText(
                apiBytes, parameters: apiParameters, output: RtxDisamOutput.Ptx);
            Check("managed API infers a single entry",
                ptxDefault.Ptx != null &&
                ptxDefault.Ptx.Contains(".entry api", StringComparison.Ordinal),
                ptxDefault.Ptx ?? "no PTX was emitted");

            SassProgram aliasProgram = SassParser.ParseLines(new[]
            {
                ".target sm_101",
                ".section .text.alias,\"ax\",@progbits",
                ".global alias",
                "alias:",
                "    MOV R0, 0x1;",
                "    EXIT;",
            }, "<alias>");
            SassSection aliasSection = aliasProgram.Sections.First(
                section => section.Name == ".text.alias");
            string aliasPtx = SassToPtx.Translate(
                aliasSection, "alias", Array.Empty<PtxParameter>(), "sm_101");
            Check("PTX canonicalizes legacy sm_101 target",
                aliasPtx.Contains(".target sm_110", StringComparison.Ordinal) &&
                aliasPtx.Contains(".version 9.0", StringComparison.Ordinal) &&
                !aliasPtx.Contains(".target sm_101", StringComparison.Ordinal),
                aliasPtx);

            RtxDisamText both = RtxDisamApi.ContainerToText(
                apiCubinPath, "api", apiParameters, RtxDisamOutput.Both);
            Check("managed API path output",
                both.Sass != null && both.Ptx != null &&
                both.Sass.Contains(".text.api", StringComparison.Ordinal) &&
                both.Ptx.Contains(".entry api", StringComparison.Ordinal),
                "path API did not return both outputs");

            bool missingEntryRejected = false;
            try
            {
                RtxDisamApi.ContainerToText(
                    apiBytes, "missing", apiParameters, RtxDisamOutput.Ptx);
            }
            catch (InvalidDataException)
            {
                missingEntryRejected = true;
            }
            Check("managed API rejects missing entry", missingEntryRejected);
        }
        finally
        {
            try { Directory.Delete(apiDir, true); } catch { }
        }
    }
}
