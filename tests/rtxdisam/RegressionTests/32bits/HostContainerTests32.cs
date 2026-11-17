using System.Text;
using nvDismDecode;
using rtxdisam;
using static RegressionTests.TestContext;
using static RegressionTests.Fixtures;

namespace RegressionTests;

internal static class HostContainerTests32
{
    public static void Run()
    {
        byte[] uncompressed = BuildFatbin(1, Encoding.ASCII.GetBytes("abcabcabcabc"), 0x41, 0);

        byte[] pe32 = BuildPe32(uncompressed);
        var peBlobs = PeFatbinReader.FindEmbeddedFatbins(pe32);
        Check("pe32 fatbin count", peBlobs.Count == 1, $"count={peBlobs.Count}");
        Check("pe32 fatbin parses", peBlobs.Count == 1 && FatbinFile.Parse(peBlobs[0]).Entries.Count == 1);

        byte[] coff32 = BuildCoff32(uncompressed);
        var coffBlobs = CoffFatbinReader.FindEmbeddedFatbins(coff32);
        Check("coff32 (i386) fatbin count", coffBlobs.Count == 1, $"count={coffBlobs.Count}");
        Check("coff32 (i386) fatbin parses",
            coffBlobs.Count == 1 && FatbinFile.Parse(coffBlobs[0]).Entries.Count == 1);

        byte[] elf32 = BuildElf32WithFatbin(uncompressed);
        CubinFile cubin32 = CubinFile.Load(elf32);
        Check("elf32 embedded fatbin count", cubin32.EmbeddedFatbins.Count == 1,
            $"count={cubin32.EmbeddedFatbins.Count}");
        Check("elf32 embedded fatbin parses",
            cubin32.EmbeddedFatbins.Count == 1 &&
            FatbinFile.Parse(cubin32.EmbeddedFatbins[0]).Entries.Count == 1);

        byte[] renamedElf32 = BuildElf32WithFatbin(uncompressed, ".rodata");
        CubinFile renamedCubin32 = CubinFile.Load(renamedElf32);
        Check("elf32 renamed-section fallback", renamedCubin32.EmbeddedFatbins.Count == 1,
            $"count={renamedCubin32.EmbeddedFatbins.Count}");

        byte[] pe32Wrapper = BuildPe32WithWrapper(uncompressed);
        var pe32WrapperBlobs = PeFatbinReader.FindEmbeddedFatbins(pe32Wrapper);
        Check("pe32 wrapper count", pe32WrapperBlobs.Count == 1, $"count={pe32WrapperBlobs.Count}");
        Check("pe32 wrapper parses",
            pe32WrapperBlobs.Count == 1 &&
            FatbinFile.Parse(pe32WrapperBlobs[0]).Entries.Count == 1);

        byte[] wrapperElf32 = BuildElf32WithWrapper(uncompressed);
        CubinFile wrapperCubin32 = CubinFile.Load(wrapperElf32);
        Check("elf32 .nvFatBinSegment wrapper", wrapperCubin32.EmbeddedFatbins.Count == 1,
            $"count={wrapperCubin32.EmbeddedFatbins.Count}");
        Check("elf32 wrapper payload parses",
            wrapperCubin32.EmbeddedFatbins.Count == 1 &&
            FatbinFile.Parse(wrapperCubin32.EmbeddedFatbins[0]).Entries.Count == 1);

        CubinFile cuinfo32 = CubinFile.Load(BuildElf32WithCuinfo(86));
        Check("elf32 cuinfo arch 86", cuinfo32.Architecture == 86, $"arch={cuinfo32.Architecture}");
        Check("elf32 cuinfo is device image", cuinfo32.IsDeviceImage);
        Check("elf32 cuinfo tool version", cuinfo32.Notes.CuinfoToolVersion == 0x86,
            $"version={cuinfo32.Notes.CuinfoToolVersion}");

        CubinFile tkinfo32 = CubinFile.Load(BuildElf32WithTkinfo());
        Check("elf32 tkinfo first note", tkinfo32.Notes.TkinfoPresent &&
            tkinfo32.Notes.TkinfoToolName == "nvlink" &&
            tkinfo32.Notes.TkinfoVersion == 2,
            $"tool={tkinfo32.Notes.TkinfoToolName}");

        CubinFile reloc32 = CubinFile.Load(BuildElf32Reloc());
        Check("elf32 reloc section count", reloc32.Sections.Count == 1, $"count={reloc32.Sections.Count}");
        if (reloc32.Sections.Count == 1)
        {
            CubinSection relocSection = reloc32.Sections[0];
            Check("elf32 reloc count", relocSection.Relocations.Count == 3, $"count={relocSection.Relocations.Count}");
            if (relocSection.Relocations.Count == 3)
            {
                Check("elf32 reloc types",
                    relocSection.Relocations[0].Type == 56 &&
                    relocSection.Relocations[1].Type == 57 &&
                    relocSection.Relocations[2].Type == 58);
                Check("elf32 reloc symbol names",
                    relocSection.Relocations[0].SymbolName == "f" &&
                    relocSection.Relocations[1].SymbolName == "f__0" &&
                    relocSection.Relocations[2].SymbolName == "g");
            }
            Check("elf32 reloc target dedup",
                relocSection.RelocationTargets.Count == 1 && relocSection.RelocationTargets[0] == 0x20);
            Check("elf32 duplicate function suffix",
                relocSection.Functions.Count == 3 &&
                relocSection.Functions[0].Name == "f" &&
                relocSection.Functions[1].Name == "f__0" &&
                relocSection.Functions[1].OriginalName == "f" &&
                relocSection.Functions[2].Name == "g");
        }

        CubinFile brx32 = CubinFile.Load(BuildElf32Brx());
        Check("elf32 brx info section parsed", brx32.Sections.Count == 1, $"count={brx32.Sections.Count}");
        if (brx32.Sections.Count == 1)
        {
            CubinSection brxSection = brx32.Sections[0];
            Check("elf32 brx indirect targets",
                brxSection.IndirectBranchTargets.TryGetValue(0x10, out List<ulong>? brxTargets) &&
                brxTargets.Count == 3 &&
                brxTargets[0] == 0x100 && brxTargets[1] == 0x120 && brxTargets[2] == 0x140);
            Check("elf32 info stack offset attribute", brxSection.HasStackOffset);
        }

        // Managed/API-oriented extraction flow from 32-bit host containers:
        // embed a fatbin (containing a cubin) in PE32/ELF32 hosts, extract via
        // host readers, and verify RtxDisamApi can process the extracted payload.
        string apiDir = Path.Combine(Path.GetTempPath(), "rtxdisam-api32-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(apiDir);
        try
        {
            string sassPath = Path.Combine(apiDir, "api32.sass");
            string cubinPath = Path.Combine(apiDir, "api32.cubin");
            File.WriteAllText(sassPath, string.Join("\n", new[]
            {
                ".target sm_75",
                ".section .text.api32,\"ax\",@progbits",
                ".global api32",
                "api32:",
                "    MOV R1, c[0x0][0x28];",
                "    EXIT;",
            }));
            SassAssembler.Assemble(sassPath, "sm_75", cubinPath);
            byte[] cubinBytes = File.ReadAllBytes(cubinPath);

            string fatbinPath = Path.Combine(apiDir, "api32.fatbin");
            FatbinWriter.Write(fatbinPath, cubinBytes, CubinFile.Load(cubinBytes).Architecture);
            byte[] fatbinBytes = File.ReadAllBytes(fatbinPath);

            foreach (var (label, hostBytes) in new (string, byte[])[]
            {
                ("pe32", BuildPe32(fatbinBytes)),
                ("elf32", BuildElf32WithFatbin(fatbinBytes)),
            })
            {
                var blobs = label == "pe32"
                    ? PeFatbinReader.FindEmbeddedFatbins(hostBytes)
                    : CubinFile.Load(hostBytes).EmbeddedFatbins;
                Check($"32-bit {label} host->fatbin extraction count", blobs.Count == 1, $"count={blobs.Count}");
                if (blobs.Count == 1)
                {
                    RtxDisamText sass = RtxDisamApi.ContainerToText(blobs[0], output: RtxDisamOutput.Sass);
                    Check($"32-bit {label} extracted fatbin API SASS",
                        sass.Sass != null && sass.Sass.Contains(".text.api32", StringComparison.Ordinal),
                        sass.Sass ?? "<null>");
                }
            }
        }
        finally
        {
            try { Directory.Delete(apiDir, true); } catch { }
        }
    }
}
