using System.Buffers.Binary;
using System.Text;
using nvDismDecode;
using rtxdisam;
using static RegressionTests.TestContext;

namespace RegressionTests;

internal static class FatbinPtx32BitTests
{
    public static void Run()
    {
        string dir = Path.Combine(Path.GetTempPath(), "rtxdisam-fatbin32-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // 32-bit address-space PTX fatbin (relevant for 32-bit ACuda/PhysX flows).
            string ptx32Path = Path.Combine(dir, "sample32.ptx");
            string ptx32FatbinPath = Path.Combine(dir, "sample32-ptx.fatbin");
            File.WriteAllText(ptx32Path, string.Join("\n", new[]
            {
                ".version 4.1",
                ".target sm_52",
                ".address_size 32",
                ".visible .entry ptxk32(.param .u32 p0, .param .u32 p1)",
                "{",
                "  .reg .u32 %r<3>;",
                "  ld.param.u32 %r1, [p0];",
                "  ld.param.u32 %r2, [p1];",
                "  st.global.u32 [%r1], %r2;",
                "  ret;",
                "}",
            }));
            FatbinWriter.Write(
                ptx32FatbinPath,
                new[]
                {
                    new FatbinImage(File.ReadAllBytes(ptx32Path), 52, FatbinImageKind.Ptx),
                });
            byte[] ptx32FatbinBytes = File.ReadAllBytes(ptx32FatbinPath);
            FatbinEntry ptx32Entry = FatbinFile.Parse(ptx32FatbinBytes).Entries[0];
            Check("fatbin writer 32-bit PTX kind", ptx32Entry.IsPtx);
            Check("fatbin writer 32-bit PTX code version",
                ptx32Entry.CodeVersionMajor == 4 && ptx32Entry.CodeVersionMinor == 1,
                $"ver={ptx32Entry.CodeVersionMajor}.{ptx32Entry.CodeVersionMinor}");
            string ptx32Text = Encoding.UTF8.GetString(ptx32Entry.Payload);
            Check("fatbin writer 32-bit PTX payload",
                ptx32Text.Contains(".address_size 32", StringComparison.Ordinal) &&
                ptx32Text.Contains(".entry ptxk32", StringComparison.Ordinal),
                ptx32Text.Length > 120 ? ptx32Text[..120] : ptx32Text);

            // CUDA 10.2 toolkit PTX packaging for 32-bit address-space kernels.
            string ptx32_102Path = Path.Combine(dir, "sample32-102.ptx");
            string ptx32_102FatbinPath = Path.Combine(dir, "sample32-102.fatbin");
            File.WriteAllText(ptx32_102Path, string.Join("\n", new[]
            {
                ".version 4.1",
                ".target sm_52",
                ".address_size 32",
                ".visible .entry ptxk32()",
                "{",
                "  ret;",
                "}",
            }));
            FatbinWriter.Write(
                ptx32_102FatbinPath,
                new[]
                {
                    new FatbinImage(File.ReadAllBytes(ptx32_102Path), 52, FatbinImageKind.Ptx),
                },
                toolkit: FatbinToolkit.Cuda102);
            byte[] ptx32_102FatbinBytes = File.ReadAllBytes(ptx32_102FatbinPath);
            FatbinEntry ptx32_102Entry = FatbinFile.Parse(ptx32_102FatbinBytes).Entries[0];
            Check("fatbin writer CUDA 10.2 32-bit PTX kind", ptx32_102Entry.IsPtx);
            Check("fatbin writer CUDA 10.2 32-bit PTX header/code version",
                BinaryPrimitives.ReadUInt32LittleEndian(ptx32_102FatbinBytes.AsSpan(16 + 4)) == 72 &&
                BinaryPrimitives.ReadUInt32LittleEndian(ptx32_102FatbinBytes.AsSpan(16 + 20)) == 64 &&
                ptx32_102Entry.CodeVersionMajor == 4 &&
                ptx32_102Entry.CodeVersionMinor == 1,
                $"hdr={BinaryPrimitives.ReadUInt32LittleEndian(ptx32_102FatbinBytes.AsSpan(16 + 4))} ver={ptx32_102Entry.CodeVersionMajor}.{ptx32_102Entry.CodeVersionMinor}");
            Check("fatbin writer CUDA 10.2 32-bit PTX payload",
                Encoding.UTF8.GetString(ptx32_102Entry.Payload).Contains(".address_size 32", StringComparison.Ordinal));

            // Managed API extraction/scanning flow: 32-bit PTX fatbin embedded in 32-bit host containers.
            byte[] hostPe32 = Fixtures.BuildPe32(ptx32FatbinBytes);
            var peBlobs = PeFatbinReader.FindEmbeddedFatbins(hostPe32);
            Check("pe32 host embedded 32-bit PTX fatbin count", peBlobs.Count == 1, $"count={peBlobs.Count}");
            if (peBlobs.Count == 1)
            {
                FatbinEntry e = FatbinFile.Parse(peBlobs[0]).Entries[0];
                Check("pe32 host extracted 32-bit PTX fatbin", e.IsPtx &&
                    Encoding.UTF8.GetString(e.Payload).Contains(".address_size 32", StringComparison.Ordinal));
            }

            byte[] hostElf32 = Fixtures.BuildElf32WithFatbin(ptx32FatbinBytes);
            CubinFile hostCubin = CubinFile.Load(hostElf32);
            Check("elf32 host embedded 32-bit PTX fatbin count", hostCubin.EmbeddedFatbins.Count == 1,
                $"count={hostCubin.EmbeddedFatbins.Count}");
            if (hostCubin.EmbeddedFatbins.Count == 1)
            {
                FatbinEntry e = FatbinFile.Parse(hostCubin.EmbeddedFatbins[0]).Entries[0];
                Check("elf32 host extracted 32-bit PTX fatbin", e.IsPtx &&
                    Encoding.UTF8.GetString(e.Payload).Contains(".address_size 32", StringComparison.Ordinal));
            }

            // 32-bit param semantic: cubin assembled with toolkit 10.2/sm_50, exposed through API
            // and 32-bit host extraction (parity with 64-bit managed API section).
            string sassPath = Path.Combine(dir, "api32sem.sass");
            string cubinPath = Path.Combine(dir, "api32sem.cubin");
            File.WriteAllText(sassPath, string.Join("\n", new[]
            {
                ".target sm_50",
                ".section .text.api32sem,\"ax\",@progbits",
                ".global api32sem",
                "api32sem:",
                "    MOV R0, c[0x0][0x140];",
                "    MOV32I R1, 0x144;",
                "    LDG.E R2, [R0];",
                "    STG.E [R1], R2;",
                "    EXIT;",
            }));
            SassAssembler.Assemble(sassPath, "sm_50", cubinPath, null, "10.2");
            byte[] cubinBytes = File.ReadAllBytes(cubinPath);
            var p32 = new[] { new PtxParameter("out", "u32", 0x140, 4) };
            RtxDisamText cubinPtx = RtxDisamApi.ContainerToText(cubinBytes, "api32sem", p32, RtxDisamOutput.Ptx);
            Check("32-bit param cubin API PTX output",
                cubinPtx.Ptx != null &&
                cubinPtx.Ptx.Contains(".entry api32sem", StringComparison.Ordinal) &&
                cubinPtx.Ptx.Contains(".param .u32 out", StringComparison.Ordinal),
                cubinPtx.Ptx ?? "<null>");

            string fatbinCubinPath = Path.Combine(dir, "api32sem.fatbin");
            FatbinWriter.Write(fatbinCubinPath, cubinBytes, CubinFile.Load(cubinBytes).Architecture);
            byte[] fatbinCubinBytes = File.ReadAllBytes(fatbinCubinPath);
            byte[] pe32Host = Fixtures.BuildPe32(fatbinCubinBytes);
            var pe32Blobs = PeFatbinReader.FindEmbeddedFatbins(pe32Host);
            Check("pe32 host extracted 32-bit-param cubin fatbin count", pe32Blobs.Count == 1, $"count={pe32Blobs.Count}");
            if (pe32Blobs.Count == 1)
            {
                RtxDisamText sass = RtxDisamApi.ContainerToText(pe32Blobs[0], output: RtxDisamOutput.Sass);
                RtxDisamText ptx = RtxDisamApi.ContainerToText(pe32Blobs[0], "api32sem", p32, RtxDisamOutput.Ptx);
                Check("32-bit host extracted cubin fatbin API SASS/PTX",
                    sass.Sass != null && sass.Sass.Contains(".text.api32sem", StringComparison.Ordinal) &&
                    ptx.Ptx != null && ptx.Ptx.Contains(".param .u32 out", StringComparison.Ordinal),
                    $"sass={sass.Sass != null} ptx={ptx.Ptx}");
            }
        }
        finally
        {
            try { Directory.Delete(dir, true); } catch { }
        }
    }
}
