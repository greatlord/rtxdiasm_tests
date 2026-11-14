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

internal static class DecoderIsaTests
{
    public static void Run()
    {
        // -- decoder regression cases -------------------------------------------------

        Equal("imad.mov",
            "IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]",
            Decode("000fe400078e00ff00000a00ff017624").Text);

        Equal("ldg",
            "LDG.E R7, [R2.64+0x20]",
            Decode("000ea2000c1e19000000200402077981").Text);

        Equal("exit", "EXIT", Decode("000fea0003800000000000000000794d").Text);

        Equal("bra", "BRA 0xc0", Decode("000fc0000383fffffffffff000007947", 0xc0).Text);

        Equal("ret.rel",
            "RET.REL.NODEC R6 0x0",
            Decode("000fea0003c3fffffffff63006007950", 0x9c0).Text);

        Equal("ret.abs zero",
            "RET.ABS.NODEC R20 0x0",
            Decode("0000000003e000000000000014007950", 0x110).Text);

        Equal("ret.abs offset",
            "RET.ABS R134 0x20",
            Decode("0000000003a000000000002086007950", 0x47c0).Text);

        Equal("guard !PT",
            "@!PT LDS RZ, [RZ]",
            Decode("000fe2000000080000000000fffff984").Text);

        Equal("imad shift exception",
            "IMAD.U32 R6, R6, 0x10000, RZ",
            Decode("000fca00078e00ff0001000006067824").Text);

        Equal("imad large shift",
            "IMAD.SHL.U32 R8, R8, 0x1000000, RZ",
            Decode("000fe200078e00ff0100000008087824").Text);

        Equal("imad negative shift",
            "IMAD.U32 R6, R6, -0x80000000, RZ",
            Decode("000fca00078e00ff8000000006067824").Text);

        Equal("imad rrb immediate",
            "IMAD R44, R4, R5, 0x1",
            Decode("000fe200078e020500000001042c7424").Text);

        Equal("depbar bitset",
            "DEPBAR.LE SB0, 0x0, {4,3,2,1}",
            Decode("000fce00000000000000801e0000791a").Text);

        Equal("depbar bitset short",
            "DEPBAR.LE SB0, 0x0, {4,3,2}",
            Decode("000fce00000000000000801c0000791a").Text);

        foreach (string architecture in new[] { "sm_70", "sm_72" })
        {
            InstructionDecoder.UseArchitecture(architecture);
            Check($"{architecture} table architecture",
                InstructionDecoder.Table.Architecture == architecture,
                InstructionDecoder.Table.Architecture);
        }
        Check("register count estimator basic", RegisterCountEstimator.Estimate(new[]
        {
            "MOV R1, c[0x0][0x28]",
        }) == 4);
        Check("register count estimator 64-bit pair", RegisterCountEstimator.Estimate(new[]
        {
            "LDG.E R4, [R22.64]",
        }) == 26);
        Check("register count estimator 128-bit vector", RegisterCountEstimator.Estimate(new[]
        {
            "LDG.E.128 R24, [R2.64]",
        }) == 30);
        Check("register count estimator return pair", RegisterCountEstimator.Estimate(new[]
        {
            "RET.ABS.NODEC R20 0x0",
        }) == 24);

        Check("spill local parser guard/width",
            rtxdisam.Program.TryGetR1Local(
                "@P0 STL.64 [R1+0x8], R2", out string spillKey,
                out int spillWidth, out bool spillStore) &&
            spillKey == "+0x8" && spillWidth == 64 && spillStore);
        Check("spill local ignores R2",
            !rtxdisam.Program.TryGetR1Local(
                "LDL R2, [R2+0x0]", out _, out _, out _));

        CubinSection SpillSection()
        {
            var section = new CubinSection
            {
                Name = ".text.spilltest",
                DisplayName = ".text.spilltest",
                Address = 0,
            };
            section.Functions.Add(new CubinSymbol("spilltest", 2, 0, 0, 0));
            return section;
        }
        List<(int Offset, ulong Address, DecodedInstruction Instruction)> SpillItems(
            params (string Text, ulong Address)[] texts) =>
            texts.Select(item => (
                (int)item.Address,
                item.Address,
                new DecodedInstruction { Text = item.Text, Address = item.Address }))
                .ToList();

        CubinSection lastUseSection = SpillSection();
        var lastUse = rtxdisam.Program.BuildNonSpillSlots(lastUseSection, SpillItems(
            ("LDL.LU R2, [R1+0x0]", 0)));
        Check("spill last-use is not non-spill",
            !lastUse.Contains((0UL, "+0x0")));

        CubinSection plainLoadSection = SpillSection();
        var plainLoad = rtxdisam.Program.BuildNonSpillSlots(plainLoadSection, SpillItems(
            ("LDL R2, [R1+0x4]", 0)));
        Check("plain local load is non-spill",
            plainLoad.Contains((0UL, "+0x4")));

        CubinSection vectorSection = SpillSection();
        var vector = rtxdisam.Program.BuildNonSpillSlots(vectorSection, SpillItems(
            ("STL.128 [R1+0x8], R2", 0)));
        Check("vector local store is non-spill",
            vector.Contains((0UL, "+0x8")));

        CubinSection storeOnlySection = SpillSection();
        var storeOnly = rtxdisam.Program.BuildNonSpillSlots(storeOnlySection, SpillItems(
            ("STL [R1+0x10], R2", 0)));
        Check("store-only local slot is non-spill",
            storeOnly.Contains((0UL, "+0x10")));

        CubinSection adjacentSection = SpillSection();
        var adjacent = rtxdisam.Program.BuildNonSpillSlots(adjacentSection, SpillItems(
            ("STL [R1+0x20], R2", 0),
            ("LDL R3, [R1+0x20]", 16)));
        Check("adjacent single copy is forced spill",
            !adjacent.Contains((0UL, "+0x20")));

        CubinSection vectorlessSection = SpillSection();
        var vectorlessItems = Enumerable.Range(0, 10)
            .Select(index => ($"LDL R2, [R1+0x{index:x}]", (ulong)(index * 16)))
            .ToArray();
        var vectorless = rtxdisam.Program.BuildNonSpillSlots(
            vectorlessSection, SpillItems(vectorlessItems));
        Check("vectorless many-slot function is local",
            vectorless.Contains((0UL, "+0x0")) &&
            vectorless.Contains((0UL, "+0x9")));

        CubinSection balancedSection = SpillSection();
        var balancedItems = new List<(string, ulong)>();
        for (int index = 0; index < 3; index++)
        {
            balancedItems.Add(($"STL.64 [R1+0x{index * 2:x}], R2", (ulong)(index * 32)));
            balancedItems.Add(($"LDL.64 R2, [R1+0x{index * 2:x}]", (ulong)(index * 32 + 16)));
        }
        balancedItems.Add(("LDL.LU R2, [R1+0x10]", 96));
        var balanced = rtxdisam.Program.BuildNonSpillSlots(
            balancedSection, SpillItems(balancedItems.ToArray()));
        Check("balanced 64-bit group is forced spill",
            !balanced.Contains((0UL, "+0x0")) &&
            !balanced.Contains((0UL, "+0x2")) &&
            !balanced.Contains((0UL, "+0x4")));

        CubinSection RelocationSection(
            uint type, string symbol, long addend = 0, ulong offset = 0)
        {
            var section = new CubinSection { Name = ".text.reloctest", Address = 0 };
            section.Relocations.Add(new CubinRelocation
            {
                Offset = offset,
                Type = type,
                SymbolName = symbol,
                Addend = addend,
            });
            return section;
        }
        string? ApplyRelocation(
            CubinSection section, string text, bool nvJsonStyle = false) =>
            rtxdisam.Program.TryApplyRelocation(
                section, 0, text, new Dictionary<(CubinSection, ulong), string>(),
                numericTargets: nvJsonStyle, nvJsonStyle: nvJsonStyle);

        CubinSection absConstantSection = RelocationSection(64, "sym");
        Check("relocation absolute constant-bank symbol",
            ApplyRelocation(absConstantSection, "MOV R1, c[0x0][0x0]")
                ?.Contains("c[`(sym)]") == true);

        CubinSection indexedConstantSection = RelocationSection(66, "sym", 8);
        Check("relocation indexed constant-bank symbol",
            ApplyRelocation(indexedConstantSection, "LDC R1, c[0x0][R2]")
                ?.Contains("c[R2+`((sym + 0x8))]") == true);

        CubinSection sharedWindowSection = RelocationSection(74, "shrdMem32");
        Check("relocation shared-window symbol",
            ApplyRelocation(sharedWindowSection, "LDS R1, [R2]")
                ?.Contains("[R2+`(shrdMem32)]") == true);

        CubinSection loSection = RelocationSection(56, "sym", 4);
        Check("relocation lo symbol",
            ApplyRelocation(loSection, "MOV R1, 0x0")
                ?.Contains("32@lo((sym + 0x4))") == true);
        Equal("relocation lo comma spacing",
            "MOV R1, 32@lo((sym + 0x4))",
            ApplyRelocation(loSection, "MOV R1, 0x0") ?? "");

        CubinSection hiSection = RelocationSection(57, "sym", 4);
        Check("relocation hi symbol",
            ApplyRelocation(hiSection, "MOV R1, 0x0")
                ?.Contains("32@hi((sym + 0x4))") == true);

        CubinSection callSection = RelocationSection(58, "__UFT_OFFSET");
        Check("relocation register-indirect call symbol",
            ApplyRelocation(callSection, "CALL.ABS.NOINC R2")
                ?.EndsWith("`(__UFT_OFFSET)") == true);

        CubinSection sharedFullSection = RelocationSection(55, "smem_full");
        Check("relocation shared-full symbol",
            ApplyRelocation(sharedFullSection, "LDS R1, [R2]")
                ?.Contains("`(smem_full)") == true);

        CubinSection sharedFull59Section = RelocationSection(59, "smem_full");
        Check("relocation shared-full type 59",
            ApplyRelocation(sharedFull59Section, "STS [R2], R1")
                ?.Contains("`(smem_full)") == true);

        CubinSection sharedWindowAddendSection =
            RelocationSection(74, "shrdMem32", 0x10);
        Check("relocation shared-window addend",
            ApplyRelocation(sharedWindowAddendSection, "LDS R1, [R2]")
                ?.Contains("[R2+`((shrdMem32 + 0x10))]") == true);

        CubinSection jsonAbsoluteSection = RelocationSection(64, "sym");
        Check("relocation normal JSON absolute constant",
            ApplyRelocation(jsonAbsoluteSection, "MOV R1, c[0x0][0x0]", true)
                ?.Contains("c[,") == true);

        CubinSection jsonIndexedSection = RelocationSection(66, "sym", 8);
        Check("relocation normal JSON indexed constant",
            ApplyRelocation(jsonIndexedSection, "LDC R1, c[0x0][R2]", true)
                ?.Contains("c[R2+0x8]") == true);

        CubinSection directCallSection = RelocationSection(58, "target");
        Check("relocation direct call symbol",
            ApplyRelocation(directCallSection, "CALL.ABS.NOINC 0x0")
                ?.Contains("`(target)") == true);

        CubinSection absoluteCall75Section = RelocationSection(75, "target75");
        Check("relocation absolute call type 75",
            ApplyRelocation(absoluteCall75Section, "CALL.ABS.NOINC 0x0")
                ?.Contains("`(target75)") == true);

        CubinSection registerCall114Section = RelocationSection(114, "__UFT_OFFSET");
        Check("relocation register call type 114",
            ApplyRelocation(registerCall114Section, "CALL.ABS.NOINC R2")
                ?.Contains("`(__UFT_OFFSET)") == true);

        foreach (var (architecture, fileName) in new[]
        {
            ("sm_70", "isa_102_sm70.json.zst"),
            ("sm_72", "isa_102_sm72.json.zst"),
            ("sm_75", "isa_134_sm75.json.zst"),
            ("sm_80", "isa_134_sm80.json.zst"),
            ("sm_86", "isa_134_sm86,sm87,sm88.json.zst"),
            ("sm_87", "isa_134_sm86,sm87,sm88.json.zst"),
            ("sm_88", "isa_134_sm86,sm87,sm88.json.zst"),
            ("sm_89", "isa_134_sm89.json.zst"),
            ("sm_90", "isa_134_sm90,sm90a.json.zst"),
            ("sm_90a", "isa_134_sm90,sm90a.json.zst"),
            ("sm_100", "isa_134_sm100,sm100a,sm100f.json.zst"),
            ("sm_100a", "isa_134_sm100,sm100a,sm100f.json.zst"),
            ("sm_100f", "isa_134_sm100,sm100a,sm100f.json.zst"),
            ("sm_101", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_101a", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_101f", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_110", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_110a", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_110f", "isa_134_sm101,sm101a,sm101f,sm110,sm110a,sm110f.json.zst"),
            ("sm_103", "isa_134_sm103,sm103a,sm103f.json.zst"),
            ("sm_107", "isa_134_sm107,sm107a,sm107f.json.zst"),
            ("sm_120", "isa_134_sm120,sm120a,sm120f,sm121,sm121a,sm121f.json.zst"),
            ("sm_121", "isa_134_sm120,sm120a,sm120f,sm121,sm121a,sm121f.json.zst"),
        })
        {
            string? path = architecture == "sm_75"
                ? InstructionDecoder.ResolveTablePath(architecture, "13.4")
                : InstructionDecoder.ResolveTablePath(architecture);
            Check($"{architecture} table path",
                path != null && Path.GetFileName(path) == fileName,
                path ?? "<null>");
        }
        Check("toolkit 10.2 table path",
            Path.GetFileName(InstructionDecoder.ResolveTablePath(
                "sm_70", "10.2") ?? "") == "isa_102_sm70.json.zst");
        Check("sm70 table directory",
            Path.GetFileName(Path.GetDirectoryName(
                InstructionDecoder.ResolveTablePath("sm_70") ?? "")) ==
                "isa_102_json");
        Check("sm75 10.2 table directory",
            Path.GetFileName(Path.GetDirectoryName(
                InstructionDecoder.ResolveTablePath("sm_75", "10.2") ?? "")) ==
                "isa_102_json");
        Check("sm75 13.4 table directory",
            Path.GetFileName(Path.GetDirectoryName(
                InstructionDecoder.ResolveTablePath("sm_75", "13.4") ?? "")) ==
                "isa_134_json");
        Check("toolkit 13.4 table path",
            Path.GetFileName(InstructionDecoder.ResolveTablePath(
                "sm_75", "13.4") ?? "") == "isa_134_sm75.json.zst");
        Check("toolkit 10.2 sm75 table path",
            Path.GetFileName(InstructionDecoder.ResolveTablePath(
                "sm_75", "10.2") ?? "") == "isa_102_sm75.json.zst");
        Check("sm75 defaults to 13.4",
            Path.GetFileName(InstructionDecoder.ResolveTablePath(
                "sm_75") ?? "") == "isa_134_sm75.json.zst");
        Check("cmd target validation sm75 default",
            rtxdisam.Program.ValidateTarget("sm_75", null, out _));
        Check("cmd target validation sm80/10.2",
            !rtxdisam.Program.ValidateTarget("sm_80", "10.2", out _));
        Check("cmd target validation sm75/10.2",
            rtxdisam.Program.ValidateTarget("sm_75", "10.2", out _));
        Check("cmd target validation sm75/13.4",
            rtxdisam.Program.ValidateTarget("sm_75", "13.4", out _));
        Check("cmd target validation sm35/13.4",
            !rtxdisam.Program.ValidateTarget("sm_35", "13.4", out _));
        Check("toolkit normalization",
            InstructionDecoder.NormalizeToolkit("cuda10.2") == "102" &&
            InstructionDecoder.NormalizeToolkit("13.4") == "134" &&
            !InstructionDecoder.TryNormalizeToolkit("12.0", out _));
        Check("toolkit table metadata",
            InstructionDecoder.LoadArchitecture("sm_70", "10.2").Toolkit == "102" &&
            InstructionDecoder.LoadArchitecture("sm_75", "13.4").Toolkit == "134");
        Check("pre-Volta toolkit table path",
            Path.GetFileName(PreVoltaDecoder.ResolveTablePath(
                "sm_35", "10.2") ?? "") == "isa_102_sm32,sm35,sm37.json.zst");
        string toolkitAsm = Path.GetTempFileName();
        File.WriteAllText(toolkitAsm,
            ".target sm_70\n" +
            ".toolkit 10.2\n" +
            ".section .text.k,\"ax\",@progbits\n" +
            "k:\n" +
            "MOV R1, R2;\n" +
            "EXIT;\n");
        var toolkitProgram = rtxdisam.SassParser.Parse(toolkitAsm);
        Check("source toolkit parse", toolkitProgram.Toolkit == "10.2");
        string toolkitCubin = Path.ChangeExtension(toolkitAsm, ".cubin");
        rtxdisam.SassAssembler.Assemble(
            toolkitAsm, "sm_70", toolkitCubin);
        Check("source toolkit assembly",
            CubinFile.Load(toolkitCubin).Architecture == 70);
        File.Delete(toolkitAsm);
        File.Delete(toolkitCubin);
        InstructionDecoder.UseArchitecture(70);
        Equal("sm70 padding NOP",
            "NOP",
            Decode("000fe200000e00ff000000fffffff389").Text);
        Check("sm70 special register map",
            InstructionDecoder.Table.SpecialRegisters.TryGetValue(
                "33", out string? sm70Special) && sm70Special == "SR_TID.X",
            sm70Special);
        InstructionDecoder.UseArchitecture(72);
        Equal("sm72 frnd f32 default type",
            "FRND.TRUNC R0, R0",
            Decode("000e22000020d0000000000000007307").Text);
        InstructionDecoder.UseArchitecture(86);

        PreVoltaDecoder.UseArchitecture("sm_30");
        Equal("sm30 mov constant",
            "MOV R1, c[0x0][0x44]",
            PreVoltaDecoder.Decode(0x2800400110005de4).Text);
        Equal("sm30 lds zero base",
            "LDS R6, [0x8]",
            PreVoltaDecoder.Decode(0xc100000023f19c85).Text);
        byte[] sm30Tail = new byte[128];
        BinaryPrimitives.WriteUInt64LittleEndian(
            sm30Tail.AsSpan(0), 0x200000000002f307UL);
        BinaryPrimitives.WriteUInt64LittleEndian(
            sm30Tail.AsSpan(0x40), 0x4003ffffe0001de7UL);
        for (int tailOffset = 0x48; tailOffset < 0x80; tailOffset += 8)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(
                sm30Tail.AsSpan(tailOffset), 0x4000000000001de4UL);
        }
        var sm30TailDecoded = PreVoltaDecoder.DecodeSection(sm30Tail).ToList();
        Check("sm30 control-free tail",
            sm30TailDecoded.Any(item =>
                item.Address == 0x40 && item.Instruction.Mnemonic == "BRA"));
        Equal("sm30 i2f intsize",
            "I2F.F32.S32 R8, -R0",
            PreVoltaDecoder.Decode(0x1800000001221f04).Text);
        Equal("sm30 f2i intsize",
            "F2I.S32.F64.TRUNC R6, R6",
            PreVoltaDecoder.Decode(0x1406000019a19c84).Text);
        Equal("sm30 atom default size",
            "@P0 ATOM.E.ADD R11, [R2+0x28], R11",
            PreVoltaDecoder.Decode(0x547e5800a022c005).Text);
        foreach (var (atomSize, expectedAtom) in new[]
        {
            ("U32", "ATOM.E.ADD R11, [R2+0x28], R11"),
            ("64", "ATOM.E.ADD.U64 R11, [R2+0x28], R11"),
        })
        {
            var atomSection = new rtxdisam.SassSection();
            atomSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "ATOM.E.ADD." + atomSize,
                Operands = { "R11", "[R2+0x28]", "R11" },
            });
            var atomDecoded = PreVoltaDecoder
                .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_30", atomSection))
                .ToList();
            Equal($"sm30 atom {atomSize} assembly", expectedAtom,
                atomDecoded[0].Instruction.Text);
        }
        PreVoltaDecoder.UseArchitecture("sm_35");
        Equal("sm35 isetp",
            "ISETP.GT.AND P0, PT, R4, 0xa, PT",
            PreVoltaDecoder.Decode(0xb3481c00051c101d).Text);
        Equal("sm35 i2f",
            "I2F.F32.S32 R8, -R0",
            PreVoltaDecoder.Decode(0xe5c10000001ca822).Text);
        Equal("sm35 vote",
            "VOTE.ALL R0, PT, PT",
            PreVoltaDecoder.Decode(0x86c71c00001c0002).Text);
        Equal("sm35 cctl address scale",
            "CCTL.E.IV [R2+0x28]",
            PreVoltaDecoder.Decode(0x7b800000141c0816).Text);
        Equal("sm35 ldc non-zero register",
            "LDC R2, c[0x2][R0]",
            PreVoltaDecoder.Decode(0x7ca00100001c000a).Text);
        Equal("sm35 brx target offset",
            "BRX R2 -0x30",
            PreVoltaDecoder.Decode(0x12807fffe81c083c).Text);
        Equal("sm35 nop sync",
            "@P0 NOP.S",
            PreVoltaDecoder.Decode(0x8580000000403c02).Text);
        Equal("sm35 imul32i high",
            "IMUL32I.HI R3, R0, 0x4",
            PreVoltaDecoder.Decode(0x2f000000021c000e).Text);
        Equal("sm35 bra uniform",
            "@P0 BRA.U 0x388",
            PreVoltaDecoder.Decode(0x12007ff50400023c, 0x1978).Text);
        Equal("sm35 iscadd negate",
            "ISCADD R114, R66, -R120, 0x8",
            PreVoltaDecoder.Decode(0xe0c120003c1d09ca).Text);
        Equal("sm35 ffma no false negate",
            "FFMA R17, R70, R64, R17",
            PreVoltaDecoder.Decode(0xcc004400201d1846).Text);
        Equal("sm35 lop32i immediate",
            "LOP32I.AND R105, R64, 0xff800000",
            PreVoltaDecoder.Decode(0x207fc000001d01a4).Text);
        var faddImmSection = new rtxdisam.SassSection();
        faddImmSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD",
            Operands = { "R11", "R8", "1.5" },
        });
        var faddImmDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", faddImmSection))
            .ToList();
        Equal("sm35 FADD immediate assembly", "FADD R11, R8, 1.5",
            faddImmDecoded[0].Instruction.Text);
        var daddImmSection = new rtxdisam.SassSection();
        daddImmSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DADD",
            Operands = { "R6", "R6", "3.25" },
        });
        var daddImmDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", daddImmSection))
            .ToList();
        Equal("sm35 DADD immediate assembly", "DADD R6, R6, 3.25",
            daddImmDecoded[0].Instruction.Text);
        var i2fNegSection = new rtxdisam.SassSection();
        i2fNegSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F32.S32",
            Operands = { "R8", "-R0" },
        });
        byte[] i2fNegBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", i2fNegSection);
        Check("sm35 I2F negate encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                i2fNegBytes.AsSpan(8)) == 0xe5c10000001ca822UL);
        var atomDefaultSection = new rtxdisam.SassSection();
        atomDefaultSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOM.E.ADD",
            Operands = { "R11", "[R2+0x28]", "R11" },
        });
        var atomDefaultDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", atomDefaultSection))
            .ToList();
        Equal("sm35 ATOM default size", "ATOM.E.ADD R11, [R2+0x28], R11",
            atomDefaultDecoded[0].Instruction.Text);
        var constMovSection = new rtxdisam.SassSection();
        constMovSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "c[0x0][0x44]" },
        });
        var constMovDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", constMovSection))
            .ToList();
        Equal("sm35 MOV constant assembly", "MOV R1, c[0x0][0x44]",
            constMovDecoded[0].Instruction.Text);
        var ldUnsignedSection = new rtxdisam.SassSection();
        ldUnsignedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E.U8",
            Operands = { "R6", "[R2+0x12]" },
        });
        ldUnsignedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E.U16",
            Operands = { "R0", "[R2+0x10]" },
        });
        var ldUnsignedDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", ldUnsignedSection))
            .ToList();
        Equal("sm35 LD.E.U8 assembly", "LD.E.U8 R6, [R2+0x12]",
            ldUnsignedDecoded[0].Instruction.Text);
        Equal("sm35 LD.E.U16 assembly", "LD.E.U16 R0, [R2+0x10]",
            ldUnsignedDecoded[1].Instruction.Text);
        var stsZeroSection = new rtxdisam.SassSection();
        stsZeroSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[0x10]", "R6" },
        });
        var stsZeroDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", stsZeroSection))
            .ToList();
        Equal("sm35 STS zero-base assembly", "STS [0x10], R6",
            stsZeroDecoded[0].Instruction.Text);
        var cctlIvSection = new rtxdisam.SassSection();
        cctlIvSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CCTL.E.IV",
            Operands = { "[R2+0x28]" },
        });
        var cctlIvDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", cctlIvSection))
            .ToList();
        Equal("sm35 CCTL.E.IV assembly", "CCTL.E.IV [R2+0x28]",
            cctlIvDecoded[0].Instruction.Text);
        Equal("sm35 brk default cc",
            "BRK",
            PreVoltaDecoder.Decode(0x1a000000001c003c).Text);
        byte[] sm35Bundle = new byte[64];
        BinaryPrimitives.WriteUInt64LittleEndian(sm35Bundle.AsSpan(8), 0x64c03c00089c0006);
        var sm35Decoded = PreVoltaDecoder.DecodeSection(sm35Bundle).ToList();
        Check("sm35 kepler bundle slot count", sm35Decoded.Count == 7);
        Check("sm35 kepler bundle addresses",
            sm35Decoded[0].Address == 8 && sm35Decoded[6].Address == 0x38);
        Check("pre-volta SM32/35/37 shared table",
            Path.GetFileName(PreVoltaDecoder.ResolveTablePath("sm_35")) ==
            "isa_102_sm32,sm35,sm37.json.zst");
        Check("pre-volta SM52/53 shared table",
            Path.GetFileName(PreVoltaDecoder.ResolveTablePath("sm_53")) ==
            "isa_102_sm52,sm53.json.zst");
        Check("pre-volta SM60/62 shared table",
            Path.GetFileName(PreVoltaDecoder.ResolveTablePath("sm_62")) ==
            "isa_102_sm60,sm62.json.zst");
        foreach (string preVoltaArch in new[]
        {
            "sm_30", "sm_32", "sm_35", "sm_37", "sm_50",
            "sm_52", "sm_53", "sm_60", "sm_61", "sm_62",
        })
        {
            PreVoltaDecoder.UseArchitecture(preVoltaArch);
            Check($"pre-volta table {preVoltaArch}",
                InstructionDecoder.Table.Architecture == preVoltaArch,
                InstructionDecoder.Table.Architecture);
            Check($"pre-volta special registers {preVoltaArch}",
                InstructionDecoder.Table.SpecialRegisters.Count > 0,
                InstructionDecoder.Table.SpecialRegisters.Count.ToString());
        }
        PreVoltaDecoder.UseArchitecture("sm_50");
        Equal("sm50 brk default cc",
            "BRK",
            PreVoltaDecoder.Decode(0xe34000000007000f).Text);
        Equal("sm50 false-condition branch",
            "NOP",
            PreVoltaDecoder.Decode(0xe240000000870000, 0x18).Text);
        var sm50Table = InstructionDecoder.Table;
        Check("pre-volta USCHED_INFO enum",
            sm50Table.Enums.TryGetValue("USCHED_INFO", out var usched) &&
            usched.TryGetValue("29", out string? yield) && yield == "YLD",
            usched == null ? "missing" : string.Join(",", usched.Take(3)));
        Check("pre-volta USCHED_INFO aliases",
            sm50Table.EnumTextToValue.TryGetValue("USCHED_INFO", out var uschedAliases) &&
            uschedAliases.TryGetValue("YIELD", out int yieldValue) && yieldValue == 29);
        Check("pre-volta scoreboard enum",
            sm50Table.Enums.TryGetValue("Scoreboard", out var scoreboards) &&
            scoreboards.Count == 6 && scoreboards["5"] == "SB5");
        Check("pre-volta quoted enum values",
            sm50Table.Enums.TryGetValue("ASel4", out var asel4) &&
            asel4["0"] == "0000", asel4?.GetValueOrDefault("0"));
        Check("pre-volta default-range enum",
            sm50Table.Enums.TryGetValue("B3B0", out var b3b0) &&
            b3b0["3"] == "B3", b3b0?.GetValueOrDefault("3"));
        var ffmaClass = sm50Table.Classes.FirstOrDefault(c => c.Name == "FFMA");
        Check("pre-volta control metadata present",
            ffmaClass?.Format?.Control.Count == 2 &&
            ffmaClass.Format.Control.Any(control =>
                control.Descriptors.Any(descriptor =>
                    descriptor.Type == "USCHED_INFO" &&
                    descriptor.Field == "usched_info")),
            ffmaClass?.FormatRaw);
        string? instructionType = null;
        bool hasInstructionType = ffmaClass != null &&
            ffmaClass.Properties.TryGetValue("INSTRUCTION_TYPE", out instructionType);
        Check("pre-volta class properties",
            hasInstructionType && instructionType == "INST_TYPE_COUPLED_MATH",
            instructionType);
        string? destSize = null;
        bool hasDestSize = ffmaClass != null &&
            ffmaClass.Predicates.TryGetValue("IDEST_SIZE", out destSize);
        Check("pre-volta class predicates",
            hasDestSize && destSize == "32", destSize);
        Check("pre-volta class conditions",
            ffmaClass?.ConditionsRaw.Contains("OOR_REG_ERROR") == true,
            ffmaClass?.ConditionsRaw);
        Check("pre-volta opcode names",
            ffmaClass?.OpcodeNames.Contains("FFMA") == true);
        string? maxRegs = null;
        bool hasMaxRegs = sm50Table.Parameters.TryGetValue("MAX_REG_COUNT", out maxRegs);
        Check("pre-volta parameters", hasMaxRegs && maxRegs == "255", maxRegs);
        string? itype = null;
        bool hasItype = sm50Table.Constants.TryGetValue("ITYPE_ABC_REG", out itype);
        Check("pre-volta constants", hasItype && itype == "1", itype);
        Check("pre-volta SIDL names",
            sm50Table.SidlNames.Contains("VOTE_VTG"));
        List<string>? fixLat = null;
        bool hasFixLat = sm50Table.TableSource.TryGetValue("FixLatDestMap", out fixLat);
        Check("pre-volta table source",
            hasFixLat && fixLat!.Contains("'&' 0 -> 0"),
            fixLat?.FirstOrDefault());
        string? wordSize = null;
        bool hasWordSize = sm50Table.ArchitectureInfo != null &&
            sm50Table.ArchitectureInfo.Settings.TryGetValue("word_size", out wordSize);
        Check("pre-volta architecture metadata",
            sm50Table.ArchitectureInfo?.Name == "Maxwell" &&
            hasWordSize && wordSize == "64", wordSize);
        Check("pre-volta operation properties",
            sm50Table.OperationProperties.Contains("INSTRUCTION_TYPE"));
        Check("pre-volta FUNIT metadata",
            sm50Table.Funit.Count == 1 &&
            sm50Table.Funit[0].EncodingWidth == 88 &&
            sm50Table.Funit[0].Masks.Count > 0);
        ulong nopEncoding = ulong.MaxValue;
        bool hasNopEncoding = sm50Table.NopEncoding.TryGetValue("ALL", out nopEncoding);
        Check("pre-volta NOP encoding", hasNopEncoding && nopEncoding == 0);
        var sm30Metadata = IsaTable.Load(
            PreVoltaDecoder.ResolveTablePath("sm_30")!);
        string? pm0 = null;
        string? pm7 = null;
        bool hasPm0 = sm30Metadata.SpecialRegisters.TryGetValue("4", out pm0);
        bool hasPm7 = sm30Metadata.SpecialRegisters.TryGetValue("11", out pm7);
        Check("sm30 PM special-register range",
            hasPm0 && hasPm7 && pm0 == "SR_PM0" && pm7 == "SR_PM7",
            $"{pm0}..{pm7}");
        string sm70MetadataPath = InstructionDecoder.ResolveTablePath("sm_70")!;
        var sm70Table = IsaTable.Load(sm70MetadataPath);
        string? stringMap = null;
        bool hasStringMap = sm70Table.StringMap.TryGetValue("mfuse", out stringMap);
        Check("sm70 string map", hasStringMap && stringMap == "TTUMACROFUSE", stringMap);
        Dictionary<string, int[]>? varLat = null;
        bool hasVarLat = sm70Table.Tables.TryGetValue("VarLatOperandEnc", out varLat);
        Check("sm70 source-only table", hasVarLat && varLat!.Count > 0,
            varLat?.Count.ToString());
        var encodedSection = new rtxdisam.SassSection();
        encodedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        byte[] encodedBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", encodedSection);
        Check("pre-volta encoded bundle size", encodedBytes.Length == 32);
        Check("pre-volta encoded word",
            BitConverter.ToUInt64(encodedBytes, 8) == 0x5c98078000270001UL);
        var encodedDecoded = PreVoltaDecoder.DecodeSection(encodedBytes).ToList();
        Equal("pre-volta encoded round trip",
            "MOV R1, R2",
            encodedDecoded[0].Instruction.Text);
        var branchSection = new rtxdisam.SassSection();
        branchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        branchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        branchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        branchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        branchSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        byte[] branchBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", branchSection);
        var branchDecoded = PreVoltaDecoder.DecodeSection(branchBytes).ToList();
        Check("pre-volta branch encoded",
            branchDecoded[2].Instruction.Mnemonic == "BRA");
        ulong branchControl = BitConverter.ToUInt64(branchBytes, 0);
        Check("pre-volta branch minimum stall",
            ((branchControl >> (2 * 21)) & 0xF) == 2,
            ((branchControl >> (2 * 21)) & 0xF).ToString());
        Check("pre-volta branch target",
            branchDecoded[2].Instruction.BranchTarget == 0x28);
        Check("pre-volta branch label address",
            branchDecoded[3].Address == 0x28);
        var exitStallSection = new rtxdisam.SassSection();
        exitStallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        exitStallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong exitStallControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", exitStallSection));
        Check("sm50 EXIT stall after MOV",
            ((exitStallControl >> 21) & 0xF) == 1,
            ((exitStallControl >> 21) & 0xF).ToString());
        var dependencySection = new rtxdisam.SassSection();
        dependencySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        dependencySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R1", "0x1" },
        });
        ulong dependencyControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", dependencySection));
        Check("sm50 intra-bundle dependency stall",
            ((dependencyControl >> 21) & 0xF) >= 6,
            ((dependencyControl >> 21) & 0xF).ToString());
        var keplerBranches = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", branchSection))
            .ToList();
        Check("Kepler branch encoded",
            keplerBranches.Any(item =>
                item.Instruction.Mnemonic == "BRA" &&
                item.Instruction.BranchTarget != null));
        var stackControlSection = new rtxdisam.SassSection();
        stackControlSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SSY",
            Operands = { "`.Lstack`" },
        });
        stackControlSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PBK",
            Operands = { "`.Lstack`" },
        });
        stackControlSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PCNT",
            Operands = { "`.Lstack`" },
        });
        stackControlSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".Lstack" });
        stackControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        var stackDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", stackControlSection))
            .ToList();
        Check("SM35 stack-control assembly",
            stackDecoded.Any(item => item.Instruction.Mnemonic == "SSY") &&
            stackDecoded.Any(item => item.Instruction.Mnemonic == "PBK") &&
            stackDecoded.Any(item => item.Instruction.Mnemonic == "PCNT"));
        Equal("SM35 ptxas SSY relative target",
            "SSY 0x110",
            PreVoltaDecoder.Decode(0x1480000048000000UL, 0x78).Text);
        var breakControlSection = new rtxdisam.SassSection();
        breakControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "BRK" });
        breakControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "CONT" });
        breakControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "KIL" });
        breakControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "LONGJMP" });
        breakControlSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        byte[] breakControlBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", breakControlSection);
        Check("SM35 BRK control encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                breakControlBytes.AsSpan(8)) == 0x1a000000001c003cUL);
        Check("SM35 CONT control encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                breakControlBytes.AsSpan(16)) == 0x1a800000001c003cUL);
        Check("SM35 KIL control encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                breakControlBytes.AsSpan(24)) == 0x19800000001c003cUL);
        Check("SM35 LONGJMP control encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                breakControlBytes.AsSpan(32)) == 0x18800000001c003cUL);
        var breakControlDecoded = PreVoltaDecoder
            .DecodeSection(breakControlBytes)
            .ToList();
        Check("SM35 break-control assembly",
            breakControlDecoded.Any(item => item.Instruction.Mnemonic == "BRK") &&
            breakControlDecoded.Any(item => item.Instruction.Mnemonic == "CONT") &&
            breakControlDecoded.Any(item => item.Instruction.Mnemonic == "KIL") &&
            breakControlDecoded.Any(item => item.Instruction.Mnemonic == "LONGJMP"));
        var middleBranchSection = new rtxdisam.SassSection();
        middleBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        middleBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        middleBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        middleBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        middleBranchSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        var middleBranchDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", middleBranchSection))
            .ToList();
        Check("pre-volta middle branch preserves slot",
            middleBranchDecoded.Any(item =>
                item.Instruction.Mnemonic == "BRA" &&
                item.Address == 0x10));
        string preVoltaAsm = Path.Combine(
            Path.GetTempPath(), "rtxdisam-prevolta-test.sass");
        string preVoltaCubin = Path.ChangeExtension(preVoltaAsm, ".cubin");
        File.WriteAllText(preVoltaAsm,
            ".target sm_50\n\n.section .text.k,\"ax\",@progbits\n" +
            ".global k\nk:\n        MOV R1, R2;\n        EXIT;\n");
        SassAssembler.Assemble(preVoltaAsm, "sm_50", preVoltaCubin);
        CubinFile assembledPreVolta = CubinFile.Load(preVoltaCubin);
        Check("pre-volta cubin architecture", assembledPreVolta.Architecture == 50);
        Check("pre-volta cubin ABI", assembledPreVolta.ToolkitFromAbi == "102");
        var preVoltaCubinDecoded = PreVoltaDecoder
            .DecodeSection(assembledPreVolta.Sections[0].Data, 0)
            .ToList();
        Equal("pre-volta cubin MOV",
            "MOV R1, R2",
            preVoltaCubinDecoded[0].Instruction.Text);
        Equal("pre-volta cubin EXIT",
            "EXIT",
            preVoltaCubinDecoded[1].Instruction.Text);
        string preVoltaFatbin = Path.ChangeExtension(preVoltaAsm, ".fatbin");
        FatbinWriter.Write(
            preVoltaFatbin, File.ReadAllBytes(preVoltaCubin), 50,
            FatbinCompression.None);
        FatbinFile parsedPreVoltaFatbin = FatbinFile.Parse(
            File.ReadAllBytes(preVoltaFatbin));
        Check("pre-volta fatbin entry",
            parsedPreVoltaFatbin.Entries.Count == 1 &&
            parsedPreVoltaFatbin.Entries[0].Architecture == 50 &&
            parsedPreVoltaFatbin.Entries[0].IsCubin);
        File.Delete(preVoltaFatbin);
        File.Delete(preVoltaAsm);
        File.Delete(preVoltaCubin);
        string keplerAsm = Path.Combine(
            Path.GetTempPath(), "rtxdisam-kepler-test.sass");
        string keplerCubin = Path.ChangeExtension(keplerAsm, ".cubin");
        File.WriteAllText(keplerAsm,
            ".target sm_35\n\n.section .text.k,\"ax\",@progbits\n" +
            ".global k\nk:\n        MOV R1, R2;\n        EXIT;\n");
        SassAssembler.Assemble(keplerAsm, "sm_35", keplerCubin);
        CubinFile assembledKepler = CubinFile.Load(keplerCubin);
        Check("Kepler cubin architecture", assembledKepler.Architecture == 35);
        Check("Kepler cubin ABI", assembledKepler.ToolkitFromAbi == "102");
        PreVoltaDecoder.UseArchitecture("sm_35");
        var sm30Section = new rtxdisam.SassSection();
        sm30Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        var sm30Decoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30Section))
            .ToList();
        Equal("SM30 IADD assembly", "IADD R1, R2, R3",
            sm30Decoded[0].Instruction.Text);
        var sm30MixSection = new rtxdisam.SassSection();
        sm30MixSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm30MixSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R1", "R2" },
        });
        sm30MixSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD",
            Operands = { "R4", "R1", "R2" },
        });
        sm30MixSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30MixControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30MixSection));
        Check("SM30 mixed ALU control",
            sm30MixControl == 0x200000000002f307UL,
            sm30MixControl.ToString("x16"));
        var sm30LoadAluSection = new rtxdisam.SassSection();
        sm30LoadAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R1", "[R2]" },
        });
        sm30LoadAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R1", "0x1" },
        });
        sm30LoadAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R3" },
        });
        sm30LoadAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30LoadAluControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30LoadAluSection));
        Check("SM30 load-to-ALU delay",
            ((sm30LoadAluControl >> 4) & 0xFF) == 0x3F,
            ((sm30LoadAluControl >> 4) & 0xFF).ToString("x"));
        var sm30MemorySection = new rtxdisam.SassSection();
        sm30MemorySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R1", "[R2]" },
        });
        sm30MemorySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R1" },
        });
        var sm30MemoryDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30MemorySection))
            .ToList();
        Equal("SM30 LD assembly", "LD.E R1, [R2]",
            sm30MemoryDecoded[0].Instruction.Text);
        Equal("SM30 ST assembly", "ST.E [R2], R1",
            sm30MemoryDecoded[1].Instruction.Text);
        var sm30MixedSection = new rtxdisam.SassSection();
        foreach (var (mnemonic, operands) in new[]
        {
            ("MOV", new[] { "R1", "R2" }),
            ("IADD", new[] { "R3", "R1", "R2" }),
            ("FADD", new[] { "R4", "R1", "R2" }),
            ("MUFU.RCP", new[] { "R5", "R1" }),
        })
        {
            var mixedInstruction = new rtxdisam.SassInstruction
            {
                Mnemonic = mnemonic,
            };
            foreach (string operand in operands)
                mixedInstruction.Operands.Add(operand);
            sm30MixedSection.Nodes.Add(mixedInstruction);
        }
        var sm30MixedDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30MixedSection))
            .ToList();
        Equal("SM30 mixed MOV", "MOV R1, R2",
            sm30MixedDecoded[0].Instruction.Text);
        Equal("SM30 mixed IADD", "IADD R3, R1, R2",
            sm30MixedDecoded[1].Instruction.Text);
        Equal("SM30 mixed FADD", "FADD R4, R1, R2",
            sm30MixedDecoded[2].Instruction.Text);
        Equal("SM30 mixed MUFU", "MUFU.RCP R5, R1",
            sm30MixedDecoded[3].Instruction.Text);
        var sm30BranchSection = new rtxdisam.SassSection();
        sm30BranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm30BranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30BranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        sm30BranchSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        var sm30BranchDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30BranchSection))
            .ToList();
        Check("SM30 branch assembly",
            sm30BranchDecoded.Any(item => item.Instruction.Mnemonic == "BRA"));
        PreVoltaDecoder.UseArchitecture("sm_35");
        var keplerDecoded = PreVoltaDecoder
            .DecodeSection(assembledKepler.Sections[0].Data, 0)
            .ToList();
        Equal("Kepler cubin MOV", "MOV R1, R2",
            keplerDecoded[0].Instruction.Text);
        Equal("Kepler cubin EXIT", "EXIT",
            keplerDecoded[1].Instruction.Text);
        File.Delete(keplerAsm);
        File.Delete(keplerCubin);
        foreach (string asmArch in new[]
        {
            "sm_30", "sm_32", "sm_35", "sm_50", "sm_52", "sm_53",
            "sm_60", "sm_61", "sm_62",
        })
        {
            var simpleSection = new rtxdisam.SassSection();
            simpleSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { "R1", "R2" },
            });
            byte[] simpleBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
                asmArch, simpleSection);
            var simpleDecoded = PreVoltaDecoder.DecodeSection(simpleBytes).ToList();
            Check($"pre-volta assembly {asmArch}",
                simpleDecoded.Count > 0 &&
                simpleDecoded[0].Instruction.Text == "MOV R1, R2");
        }
        var iaddSection = new rtxdisam.SassSection();
        iaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        var iaddDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", iaddSection))
            .ToList();
        Equal("pre-volta IADD assembly", "IADD R1, R2, R3",
            iaddDecoded[0].Instruction.Text);
        Equal("pre-volta LDS.U.32 assembly", "LDS.U.32 R0, [0x8]",
            PreVoltaDecoder.Decode(0xef4c10000087ff00).Text);
        Equal("pre-volta LDS.U.128 assembly", "LDS.U.128 R1, [R2]",
            PreVoltaDecoder.Decode(0xef4e100000070201).Text);
        var ldsSection = new rtxdisam.SassSection();
        ldsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R1", "[R2]" },
        });
        var ldsDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", ldsSection))
            .ToList();
        Equal("pre-volta LDS default size", "LDS R1, [R2]",
            ldsDecoded[0].Instruction.Text);
        var lds128Section = new rtxdisam.SassSection();
        lds128Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS.U.128",
            Operands = { "R0", "[0x4000]" },
        });
        var lds128Decoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", lds128Section))
            .ToList();
        Equal("sm35 LDS.U.128 assembly", "LDS.U.128 R0, [0x4000]",
            lds128Decoded[0].Instruction.Text);
        foreach (var (control, expectedWord) in new[]
        {
            ("BRK", 0xe34000000007000fUL),
            ("CONT", 0xe35000000007000fUL),
            ("KIL", 0xe33000000007000fUL),
            ("LONGJMP", 0xe31000000007000fUL),
        })
        {
            var controlSection = new rtxdisam.SassSection();
            controlSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = control,
            });
            byte[] controlBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", controlSection);
            Check($"sm50 {control} encoding",
                BinaryPrimitives.ReadUInt64LittleEndian(
                    controlBytes.AsSpan(8)) == expectedWord);
        }
        var nopSyncSection = new rtxdisam.SassSection();
        nopSyncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP.S",
            Guard = "@P0",
        });
        byte[] nopSyncBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", nopSyncSection);
        Check("pre-volta NOP.S encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                nopSyncBytes.AsSpan(8)) == 0x8580000000403c02UL);
        var brxTargetSection = new rtxdisam.SassSection();
        brxTargetSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRX",
            Operands = { "R2", "-0x30" },
        });
        byte[] brxTargetBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", brxTargetSection);
        Check("pre-volta BRX target encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                brxTargetBytes.AsSpan(0x8)) == 0x12807fffe81c083cUL);
        foreach (string aliasArch in new[] { "sm_32", "sm_35", "sm_37" })
        {
            foreach (var (alias, operands, expected) in new[]
            {
                ("LDG.CT", new[] { "R1", "[R2]" },
                    "TEX.LZ R1, R2, 0x0, 1D, 0x4"),
                ("LDG.CG", new[] { "R3", "[R4]" },
                    "TLD.LZ R3, R4, 0x0, 1D, 0x4"),
                ("LDG.CS", new[] { "R5", "[R6]" },
                    "TLD.LZ.CL R5, R6, 0x0, 1D, 0x4"),
                ("CCTLT", Array.Empty<string>(), "CCTLT.IVALL"),
            })
            {
                var aliasSection = new rtxdisam.SassSection();
                var aliasInstruction = new rtxdisam.SassInstruction
                {
                    Mnemonic = alias,
                };
                aliasInstruction.Operands.AddRange(operands);
                aliasSection.Nodes.Add(aliasInstruction);
                var aliasDecoded = PreVoltaDecoder
                    .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                        aliasArch, aliasSection))
                    .ToList();
                Equal($"pre-volta {aliasArch} {alias} alias", expected,
                    aliasDecoded[0].Instruction.Text);
            }
        }
        var isubSection = new rtxdisam.SassSection();
        isubSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R1", "R2", "R3" },
        });
        var directSubSection = new rtxdisam.SassSection();
        directSubSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "-R3" },
        });
        byte[] isubBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", isubSection);
        byte[] directSubBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", directSubSection);
        Check("pre-volta ISUB remap",
            BinaryPrimitives.ReadUInt64LittleEndian(isubBytes.AsSpan(8)) ==
            BinaryPrimitives.ReadUInt64LittleEndian(directSubBytes.AsSpan(8)));
        var negRaSection = new rtxdisam.SassSection();
        negRaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "-R2", "R3" },
        });
        var negRaDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", negRaSection))
            .ToList();
        Equal("pre-volta PSign Ra negation", "IADD R1, -R2, R3",
            negRaDecoded[0].Instruction.Text);
        var isubImmSection = new rtxdisam.SassSection();
        isubImmSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R1", "R2", "0x5" },
        });
        var isubImmDirect = new rtxdisam.SassSection();
        isubImmDirect.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "-0x5" },
        });
        Check("pre-volta ISUB immediate remap",
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isubImmSection).AsSpan(8)) ==
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isubImmDirect).AsSpan(8)));
        var isub1Section = new rtxdisam.SassSection();
        isub1Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R1", "0x5", "R2" },
        });
        var isub1Direct = new rtxdisam.SassSection();
        isub1Direct.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "-R2", "0x5" },
        });
        Check("pre-volta ISUB1 immediate remap",
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isub1Section).AsSpan(8)) ==
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isub1Direct).AsSpan(8)));
        var isubConstSection = new rtxdisam.SassSection();
        isubConstSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R1", "R2", "c[0x0][0x0]" },
        });
        var isubConstDirect = new rtxdisam.SassSection();
        isubConstDirect.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "-c[0x0][0x0]" },
        });
        byte[] isubConstBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", isubConstSection);
        Check("pre-volta ISUB constant remap",
            BinaryPrimitives.ReadUInt64LittleEndian(isubConstBytes.AsSpan(8)) ==
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isubConstDirect).AsSpan(8)));
        var isubConstDecoded = PreVoltaDecoder.DecodeSection(isubConstBytes).ToList();
        Equal("pre-volta ISUB constant text",
            "IADD R1, R2, -c[0x0][0x0]",
            isubConstDecoded[0].Instruction.Text);
        var isubConst1Section = new rtxdisam.SassSection();
        isubConst1Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R1", "c[0x0][0x0]", "R2" },
        });
        var isubConst1Direct = new rtxdisam.SassSection();
        isubConst1Direct.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "-R2", "c[0x0][0x0]" },
        });
        Check("pre-volta ISUB constant-first remap",
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isubConst1Section).AsSpan(8)) ==
            BinaryPrimitives.ReadUInt64LittleEndian(
                rtxdisam.PreVoltaAssembler.EncodeSection(
                    "sm_50", isubConst1Direct).AsSpan(8)));
        var xmadSection = new rtxdisam.SassSection();
        xmadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R1", "R2", "R3", "R4" },
        });
        var xmadDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", xmadSection))
            .ToList();
        Equal("pre-volta XMAD assembly", "XMAD R1, R2, R3, R4",
            xmadDecoded[0].Instruction.Text);
        var faddSection = new rtxdisam.SassSection();
        faddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD",
            Operands = { "R1", "R2", "R3" },
        });
        var faddDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", faddSection))
            .ToList();
        Equal("pre-volta FADD assembly", "FADD R1, R2, R3",
            faddDecoded[0].Instruction.Text);
        var isetpSection = new rtxdisam.SassSection();
        isetpSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GT.AND",
            Operands = { "P0", "PT", "R1", "R2", "PT" },
        });
        var isetpDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", isetpSection))
            .ToList();
        Equal("pre-volta ISETP assembly",
            "ISETP.GT.AND P0, PT, R1, R2, PT",
            isetpDecoded[0].Instruction.Text);
        var iaddNegSection = new rtxdisam.SassSection();
        iaddNegSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R0", "-R0", "R7" },
        });
        var iaddNegDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", iaddNegSection))
            .ToList();
        Equal("pre-volta IADD negate assembly",
            "IADD R0, -R0, R7",
            iaddNegDecoded[0].Instruction.Text);
        var iscaddSection = new rtxdisam.SassSection();
        iscaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISCADD",
            Operands = { "R114", "R66", "-R120", "0x8" },
        });
        var iscaddDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", iscaddSection))
            .ToList();
        Equal("pre-volta ISCADD assembly",
            "ISCADD R114, R66, -R120, 0x8",
            iscaddDecoded[0].Instruction.Text);
        var ffmaSection = new rtxdisam.SassSection();
        ffmaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FFMA",
            Operands = { "R17", "R70", "R64", "R17" },
        });
        var ffmaDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", ffmaSection))
            .ToList();
        Equal("pre-volta FFMA assembly",
            "FFMA R17, R70, R64, R17",
            ffmaDecoded[0].Instruction.Text);
        var lop32iSection = new rtxdisam.SassSection();
        lop32iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP32I.AND",
            Operands = { "R105", "R64", "0xff800000" },
        });
        var lop32iDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", lop32iSection))
            .ToList();
        Equal("pre-volta LOP32I assembly",
            "LOP32I.AND R105, R64, 0xff800000",
            lop32iDecoded[0].Instruction.Text);
        var convSection = new rtxdisam.SassSection();
        convSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F32.S32",
            Operands = { "R1", "R2" },
        });
        convSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.S32.F32.TRUNC",
            Operands = { "R3", "R1" },
        });
        var convDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", convSection))
            .ToList();
        Equal("pre-volta I2F assembly", "I2F.F32.S32 R1, R2",
            convDecoded[0].Instruction.Text);
        Equal("pre-volta F2I assembly", "F2I.S32.F32.TRUNC R3, R1",
            convDecoded[1].Instruction.Text);
        var miscSection = new rtxdisam.SassSection();
        miscSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R1", "SR_LANEID" },
        });
        miscSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VOTE.ALL",
            Operands = { "R2", "PT", "PT" },
        });
        var miscDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", miscSection))
            .ToList();
        Equal("pre-volta S2R assembly", "S2R R1, SR_LANEID",
            miscDecoded[0].Instruction.Text);
        Equal("pre-volta VOTE assembly", "VOTE.ALL R2, PT, PT",
            miscDecoded[1].Instruction.Text);
        var arithmeticSection = new rtxdisam.SassSection();
        arithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R1", "R2", "R3", "R4" },
        });
        arithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R5", "0xf" },
        });
        arithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R6", "R1", "0x3" },
        });
        arithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHR.U32",
            Operands = { "R7", "R6", "0x1" },
        });
        var arithmeticDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", arithmeticSection))
            .ToList();
        Equal("pre-volta IMAD assembly", "IMAD R1, R2, R3, R4",
            arithmeticDecoded[0].Instruction.Text);
        Equal("pre-volta MOV32I assembly", "MOV32I R5, 0xf",
            arithmeticDecoded[1].Instruction.Text);
        Equal("pre-volta SHL assembly", "SHL R6, R1, 0x3",
            arithmeticDecoded[2].Instruction.Text);
        Equal("pre-volta SHR assembly", "SHR.U32 R7, R6, 0x1",
            arithmeticDecoded[3].Instruction.Text);
        var lop3Section = new rtxdisam.SassSection();
        lop3Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP3.LUT",
            Operands = { "R4", "R2", "R3", "R1", "0x1e" },
        });
        var lop3Decoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", lop3Section))
            .ToList();
        Equal("pre-volta LOP3 assembly", "LOP3.LUT R4, R2, R3, R1, 0x1e",
            lop3Decoded[0].Instruction.Text);
        var bitSection = new rtxdisam.SassSection();
        bitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PRMT",
            Operands = { "R1", "R2", "0x5410", "R3" },
        });
        bitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BFE.U32",
            Operands = { "R4", "R2", "R3" },
        });
        bitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BFI",
            Operands = { "R5", "R2", "R3", "R1" },
        });
        var bitDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", bitSection))
            .ToList();
        Equal("pre-volta PRMT assembly", "PRMT R1, R2, 0x5410, R3",
            bitDecoded[0].Instruction.Text);
        Equal("pre-volta BFE assembly", "BFE.U32 R4, R2, R3",
            bitDecoded[1].Instruction.Text);
        Equal("pre-volta BFI assembly", "BFI R5, R2, R3, R1",
            bitDecoded[2].Instruction.Text);
        var floSection = new rtxdisam.SassSection();
        floSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FLO.U32",
            Operands = { "R3", "R4" },
        });
        var floDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", floSection))
            .ToList();
        Equal("pre-volta FLO assembly", "FLO.U32 R3, R4",
            floDecoded[0].Instruction.Text);
        var reuseSection = new rtxdisam.SassSection();
        reuseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2.reuse", "R3" },
        });
        reuseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R1", "R3.reuse" },
        });
        var reuseDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", reuseSection))
            .ToList();
        Equal("pre-volta reuse assembly", "IADD R1, R2.reuse, R3",
            reuseDecoded[0].Instruction.Text);
        Equal("pre-volta reuse operand 2 assembly", "IADD R4, R1, R3.reuse",
            reuseDecoded[1].Instruction.Text);
        var mufuSection = new rtxdisam.SassSection();
        mufuSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MUFU.RCP",
            Operands = { "R1", "R2" },
        });
        var mufuDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", mufuSection))
            .ToList();
        Equal("pre-volta MUFU assembly", "MUFU.RCP R1, R2",
            mufuDecoded[0].Instruction.Text);
        var shfSection = new rtxdisam.SassSection();
        shfSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L",
            Operands = { "R5", "R2", "0x3", "R1" },
        });
        var shfDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", shfSection))
            .ToList();
        Equal("pre-volta SHF assembly", "SHF.L R5, R2, 0x3, R1",
            shfDecoded[0].Instruction.Text);
        var memorySection = new rtxdisam.SassSection();
        memorySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        memorySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R1" },
        });
        var memoryDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", memorySection))
            .ToList();
        Equal("pre-volta LDG assembly", "LDG.E R1, [R2]",
            memoryDecoded[0].Instruction.Text);
        Equal("pre-volta STG assembly", "STG.E [R2], R1",
            memoryDecoded[1].Instruction.Text);
        var memory64Section = new rtxdisam.SassSection();
        memory64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E.64",
            Operands = { "R1", "[R2]" },
        });
        memory64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E.64",
            Operands = { "[R2]", "R1" },
        });
        var memory64Decoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", memory64Section))
            .ToList();
        Equal("pre-volta LDG 64 assembly", "LDG.E.64 R1, [R2]",
            memory64Decoded[0].Instruction.Text);
        Equal("pre-volta STG 64 assembly", "STG.E.64 [R2], R1",
            memory64Decoded[1].Instruction.Text);
        var barrierSection = new rtxdisam.SassSection();
        barrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        barrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R1", "R4" },
        });
        byte[] barrierBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", barrierSection);
        ulong automaticBarrierControl = BitConverter.ToUInt64(barrierBytes, 0);
        Check("pre-volta automatic barrier wait",
            ((automaticBarrierControl >> (21 + 11)) & 0x3F) != 0);
        var barrierDecoded = PreVoltaDecoder.DecodeSection(barrierBytes).ToList();
        Equal("pre-volta barrier load", "LDG.E R1, [R2]",
            barrierDecoded[0].Instruction.Text);
        Equal("pre-volta barrier consumer", "IADD R3, R1, R4",
            barrierDecoded[1].Instruction.Text);
        var storeBarrierSection = new rtxdisam.SassSection();
        storeBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R1" },
        });
        byte[] storeBarrierBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", storeBarrierSection);
        ulong storeBarrierControl = BitConverter.ToUInt64(storeBarrierBytes, 0);
        Check("pre-volta store without load source has no read barrier",
            ((storeBarrierControl >> 8) & 0x7) == 7 &&
            ((storeBarrierControl >> 5) & 0x7) == 7,
            $"read={(storeBarrierControl >> 8) & 0x7} write={(storeBarrierControl >> 5) & 0x7}");
        var loadStoreBarrierSection = new rtxdisam.SassSection();
        loadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "R6" },
        });
        loadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        loadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R3]", "R1" },
        });
        loadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong loadStoreBarrierControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", loadStoreBarrierSection));
        Check("pre-volta load-sourced store read barrier",
            ((loadStoreBarrierControl >> (2 * 21 + 8)) & 0x7) != 7);
        var firstLoadStoreSection = new rtxdisam.SassSection();
        firstLoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        firstLoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R1" },
        });
        firstLoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong firstLoadStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", firstLoadStoreSection));
        Check("maxwell first load/store pair",
            ((firstLoadStoreControl >> 5) & 0x7) == 7 &&
            ((firstLoadStoreControl >> (21 + 11)) & 0x3F) == 0 &&
            ((firstLoadStoreControl >> (21 + 8)) & 0x7) == 7 &&
            ((firstLoadStoreControl >> 42) & 0xF) == 2);
        var u16LoadStoreSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "c[0x0][0x20]"), ("R2", "c[0x0][0x140]"),
            ("R3", "c[0x0][0x144]"),
        })
        {
            u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E.U16",
            Operands = { "R0", "[R2]" },
        });
        u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E.U16",
            Operands = { "[R2]", "R0" },
        });
        for (int nop = 0; nop < 3; nop++)
            u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "NOP",
            });
        u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        u16LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        u16LoadStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] u16LoadStoreBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", u16LoadStoreSection);
        ulong u16LoadControl = BinaryPrimitives.ReadUInt64LittleEndian(
            u16LoadStoreBytes.AsSpan(0x20));
        ulong u16ExitControl = BinaryPrimitives.ReadUInt64LittleEndian(
            u16LoadStoreBytes.AsSpan(0x40));
        Check("maxwell U16 load/store stalls",
            (u16LoadControl & 0xF) == 2 &&
            ((u16LoadControl >> (2 * 21)) & 0xF) == 15 &&
            ((u16ExitControl >> 21) & 0xF) == 7);
        var sm50XmadStoreSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "c[0x0][0x20]"), ("R2", "c[0x0][0x140]"),
            ("R3", "c[0x0][0x144]"),
        })
        {
            sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R5", "0xf" },
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R5", "R0", "0x5", "R5" },
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.PSL",
            Operands = { "R0", "R0.H1", "0x5", "R5" },
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        for (int nop = 0; nop < 2; nop++)
            sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "NOP",
            });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm50XmadStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] sm50XmadStoreBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", sm50XmadStoreSection);
        ulong sm50XmadLoadControl = BinaryPrimitives.ReadUInt64LittleEndian(
            sm50XmadStoreBytes.AsSpan(0x20));
        ulong sm50XmadStoreControl = BinaryPrimitives.ReadUInt64LittleEndian(
            sm50XmadStoreBytes.AsSpan(0x40));
        ulong sm50XmadExitControl = BinaryPrimitives.ReadUInt64LittleEndian(
            sm50XmadStoreBytes.AsSpan(0x60));
        Check("maxwell XMAD/store sections",
            (sm50XmadLoadControl & 0xF) == 1 &&
            ((sm50XmadLoadControl >> 21) & 0xF) == 6 &&
            (sm50XmadStoreControl & 0x1FFFFF) == 0x7E2 &&
            (sm50XmadExitControl & 0xF) == 9);
        var xmadExitSection = new rtxdisam.SassSection();
        xmadExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R1", "R2", "R3", "R4" },
        });
        xmadExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong xmadExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", xmadExitSection));
        Check("maxwell XMAD/EXIT slot",
            (xmadExitControl & 0xF) == 6);
        var compareHazardSection = new rtxdisam.SassSection();
        compareHazardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GT.AND",
            Operands = { "P0", "PT", "R1", "0xa", "PT" },
        });
        compareHazardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R1", "0x1" },
        });
        compareHazardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SEL",
            Operands = { "R0", "R0", "R1", "P0" },
        });
        ulong compareHazardControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", compareHazardSection));
        Check("maxwell predicate hazard",
            ((compareHazardControl >> 21) & 0xF) == 12);
        var loadIaddChainSection = new rtxdisam.SassSection();
        loadIaddChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        loadIaddChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R0", "R0", "R3" },
        });
        loadIaddChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R4", "R0", "0x1" },
        });
        ulong loadIaddChainControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", loadIaddChainSection));
        Check("maxwell load/IADD consumed chain",
            ((loadIaddChainControl >> 21) & 0xF) == 6);
        var loadIaddExitSection = new rtxdisam.SassSection();
        loadIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        loadIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R0", "R4" },
        });
        loadIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong loadIaddExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", loadIaddExitSection));
        Check("maxwell load/IADD unconsumed",
            ((loadIaddExitControl >> 21) & 0xF) == 1);
        var firstLoadChainSection = new rtxdisam.SassSection();
        firstLoadChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        firstLoadChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R3", "[R1]" },
        });
        firstLoadChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R3" },
        });
        firstLoadChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong firstLoadChainControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", firstLoadChainSection));
        Check("maxwell first load chain barriers",
            (firstLoadChainControl & 0xF) == 6 &&
            ((firstLoadChainControl >> 5) & 0x7) == 0 &&
            ((firstLoadChainControl >> 21) & 0xF) == 1 &&
            ((firstLoadChainControl >> (21 + 5)) & 0x7) == 1 &&
            ((firstLoadChainControl >> (2 * 21 + 8)) & 0x7) == 2 &&
            ((firstLoadChainControl >> (2 * 21 + 11)) & 0x3F) == 2 &&
            ((firstLoadChainControl >> 42) & 0xF) == 2);
        var paddedLoadStoreBarrierSection = new rtxdisam.SassSection();
        paddedLoadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        paddedLoadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R3]", "R1" },
        });
        paddedLoadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        paddedLoadStoreBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong paddedLoadStoreBarrierControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", paddedLoadStoreBarrierSection));
        Check("pre-volta NOP-padded store has no read barrier",
            ((paddedLoadStoreBarrierControl >> (21 + 8)) & 0x7) == 7);
        var terminalStoreSection = new rtxdisam.SassSection();
        terminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R1" },
        });
        terminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong terminalStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", terminalStoreSection));
        Check("terminal store omits read barrier",
            ((terminalStoreControl >> 8) & 0x7) == 7 &&
            (terminalStoreControl & 0xF) == 1,
            $"read={(terminalStoreControl >> 8) & 0x7} stall={terminalStoreControl & 0xF}");
        var chainedLoadSection = new rtxdisam.SassSection();
        chainedLoadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        chainedLoadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R3", "[R1]" },
        });
        byte[] chainedLoadBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", chainedLoadSection);
        ulong chainedLoadControl = BitConverter.ToUInt64(chainedLoadBytes, 0);
        Check("pre-volta chained load barriers",
            ((chainedLoadControl >> (21 + 11)) & 0x3F) != 0 &&
            ((chainedLoadControl >> (21 + 5)) & 0x7) != 7);
        var loadBarrier6Section = new rtxdisam.SassSection();
        loadBarrier6Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "R6" },
        });
        loadBarrier6Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        loadBarrier6Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R3", "R1", "R4" },
        });
        ulong loadBarrier6Control = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", loadBarrier6Section));
        Check("maxwell load barrier 6",
            ((loadBarrier6Control >> (21 + 5)) & 0x7) == 5 &&
            ((loadBarrier6Control >> (2 * 21 + 11)) & 0x3F) == 0x20 &&
            ((loadBarrier6Control >> 21) & 0xF) == 2 &&
            ((loadBarrier6Control >> 42) & 0xF) == 2);
        var s2rBarrierSection = new rtxdisam.SassSection();
        s2rBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R1", "SR_TID.X" },
        });
        s2rBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R2", "R1", "0x2" },
        });
        s2rBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong s2rBarrierControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", s2rBarrierSection));
        Check("maxwell S2R barrier",
            ((s2rBarrierControl >> 5) & 0x7) == 0 &&
            ((s2rBarrierControl >> (21 + 11)) & 0x3F) == 1 &&
            (s2rBarrierControl & 0xF) == 15);
        var unusedS2rSection = new rtxdisam.SassSection();
        unusedS2rSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R1", "SR_LANEID" },
        });
        unusedS2rSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong unusedS2rControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", unusedS2rSection));
        Check("maxwell unused S2R has no barrier",
            ((unusedS2rControl >> 5) & 0x7) == 7 &&
            (unusedS2rControl & 0xF) == 6);
        var plainShiftSection = new rtxdisam.SassSection();
        plainShiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R1", "R2", "0x3" },
        });
        plainShiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHR.U32",
            Operands = { "R3", "R1", "0x1" },
        });
        plainShiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong plainShiftControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", plainShiftSection));
        Check("maxwell plain shift chain",
            (plainShiftControl & 0xF) == 6 &&
            ((plainShiftControl >> 21) & 0xF) == 1);
        var constantExitSection = new rtxdisam.SassSection();
        constantExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "-c[0x0][0x0]" },
        });
        constantExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong constantExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", constantExitSection));
        Check("maxwell constant IADD exit",
            ((constantExitControl >> 21) & 0xF) == 5);
        var twoIaddExitSection = new rtxdisam.SassSection();
        twoIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        twoIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R1", "R3" },
        });
        twoIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong twoIaddExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", twoIaddExitSection));
        Check("maxwell two-IADD exit",
            ((twoIaddExitControl >> 42) & 0xF) == 6);
        var selSection = new rtxdisam.SassSection();
        selSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R1", "R2", "R3", "R4" },
        });
        selSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R5", "0xf" },
        });
        selSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SEL",
            Operands = { "R6", "R1", "R5", "PT" },
        });
        selSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong selControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", selSection));
        Check("maxwell SEL stall", ((selControl >> 42) & 0xF) == 2);
        var conversionChainSection = new rtxdisam.SassSection();
        conversionChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F32.S32",
            Operands = { "R1", "R2" },
        });
        conversionChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.S32.F32.TRUNC",
            Operands = { "R3", "R1" },
        });
        conversionChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong conversionChainControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", conversionChainSection));
        Check("maxwell conversion chain",
            ((conversionChainControl >> 21) & 0xF) == 1);
        var negatedIaddExitSection = new rtxdisam.SassSection();
        negatedIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "-R3" },
        });
        negatedIaddExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong negatedIaddExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", negatedIaddExitSection));
        Check("maxwell negated IADD exit",
            ((negatedIaddExitControl >> 21) & 0xF) == 5);
        var ldsBarrierSection = new rtxdisam.SassSection();
        ldsBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R1", "[R2]" },
        });
        ldsBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R3]", "R1" },
        });
        ldsBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong ldsBarrierControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", ldsBarrierSection));
        Check("maxwell LDS barrier",
            ((ldsBarrierControl >> 5) & 0x7) == 0 &&
            ((ldsBarrierControl >> (21 + 11)) & 0x3F) == 1 &&
            ((ldsBarrierControl >> 21) & 0xF) == 1);
        var sharedStoreSection = new rtxdisam.SassSection();
        sharedStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R1", "[R2]" },
        });
        sharedStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[R2]", "R1" },
        });
        sharedStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sharedStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", sharedStoreSection));
        Check("maxwell shared store barrier",
            (sharedStoreControl & 0xF) == 6 &&
            ((sharedStoreControl >> (21 + 8)) & 0x7) == 1 &&
            ((sharedStoreControl >> (21 + 11)) & 0x3F) == 1 &&
            ((sharedStoreControl >> 42) & 0xF) == 5);
        var barSyncSection = new rtxdisam.SassSection();
        barSyncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BAR.SYNC",
            Operands = { "0x0" },
        });
        barSyncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong barSyncControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", barSyncSection));
        Check("maxwell BAR sync stall",
            (barSyncControl & 0xF) == 15);
        var shiftChainSection = new rtxdisam.SassSection();
        shiftChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R1", "SR_TID.X" },
        });
        shiftChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R2", "R1", "0x2" },
        });
        shiftChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHR",
            Operands = { "R3", "R1", "0x1e" },
        });
        shiftChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R2", "R3" },
        });
        shiftChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] shiftChainBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", shiftChainSection);
        ulong shiftChain0 = BitConverter.ToUInt64(shiftChainBytes, 0);
        ulong shiftChain1 = BitConverter.ToUInt64(shiftChainBytes, 32);
        Check("maxwell shift chain stalls",
            (shiftChain0 & 0xF) == 15 &&
            ((shiftChain0 >> 21) & 0xF) == 1 &&
            ((shiftChain0 >> 42) & 0xF) == 5 &&
            (shiftChain1 & 0xF) == 6);
        var carryChainSection = new rtxdisam.SassSection();
        carryChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1.CC", "R2", "R3" },
        });
        carryChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R4", "R5", "R6" },
        });
        carryChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R7", "[R8]" },
        });
        carryChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R9", "R10" },
        });
        carryChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] carryChainBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", carryChainSection);
        ulong carryChain0 = BitConverter.ToUInt64(carryChainBytes, 0);
        ulong carryChain1 = BitConverter.ToUInt64(carryChainBytes, 32);
        Check("maxwell carry chain stalls",
            ((carryChain0 >> 21) & 0xF) == 2 &&
            ((carryChain0 >> 42) & 0xF) == 1 &&
            (carryChain1 & 0xF) == 4);
        var independentLoadsSection = new rtxdisam.SassSection();
        independentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R6", "R7" },
        });
        independentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        independentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R3", "[R4]" },
        });
        independentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R5", "R1", "R3" },
        });
        byte[] independentLoadsBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", independentLoadsSection);
        ulong independentLoadsControl = BitConverter.ToUInt64(
            independentLoadsBytes, 0);
        ulong independentLoadsControl1 = BitConverter.ToUInt64(
            independentLoadsBytes, 32);
        Check("maxwell adjacent load barriers",
            ((independentLoadsControl >> (21 + 5)) & 0x7) == 7 &&
            ((independentLoadsControl >> (2 * 21 + 5)) & 0x7) == 5 &&
            ((independentLoadsControl1 >> 11) & 0x3F) == 0x20 &&
            ((independentLoadsControl >> 21) & 0xF) == 1 &&
            ((independentLoadsControl >> 42) & 0xF) == 2);
        var dependentLoadsSection = new rtxdisam.SassSection();
        dependentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R6", "R7" },
        });
        dependentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R1", "[R2]" },
        });
        dependentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R3", "[R1]" },
        });
        dependentLoadsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R5", "R3", "R4" },
        });
        byte[] dependentLoadsBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", dependentLoadsSection);
        ulong dependentLoadsControl = BitConverter.ToUInt64(
            dependentLoadsBytes, 0);
        ulong dependentLoadsControl1 = BitConverter.ToUInt64(
            dependentLoadsBytes, 32);
        Check("maxwell dependent load barriers",
            ((dependentLoadsControl >> (21 + 5)) & 0x7) == 5 &&
            ((dependentLoadsControl >> (2 * 21 + 5)) & 0x7) == 5 &&
            ((dependentLoadsControl1 >> 11) & 0x3F) == 0x20 &&
            ((dependentLoadsControl >> 21) & 0xF) == 2);
        var terminalBranchSection = new rtxdisam.SassSection();
        terminalBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        terminalBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        terminalBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] terminalBranchBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", terminalBranchSection);
        ulong terminalBranchControl = BitConverter.ToUInt64(
            terminalBranchBytes, 0);
        var terminalBranchDecoded = PreVoltaDecoder
            .DecodeSection(terminalBranchBytes)
            .ToList();
        Check("maxwell terminal branch keeps natural slot",
            terminalBranchDecoded.Any(item =>
                item.Instruction.Mnemonic == "BRA" && item.Address == 0x10));
        Check("maxwell terminal branch empty slot",
            ((terminalBranchControl >> 42) & 0x1FFFFF) == 0x7E0);
        var nopPaddingSection = new rtxdisam.SassSection();
        nopPaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        nopPaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        nopPaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        ulong nopPaddingControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", nopPaddingSection));
        Check("maxwell NOP sections",
            (nopPaddingControl & 0x1FFFFF) == 0x7EF &&
            ((nopPaddingControl >> 21) & 0x1FFFFF) == 0x7EF &&
            ((nopPaddingControl >> 42) & 0x1FFFFF) == 0x7EF);
        var paddedTerminalStoreSection = new rtxdisam.SassSection();
        paddedTerminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R1" },
        });
        paddedTerminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        paddedTerminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP",
        });
        paddedTerminalStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong paddedTerminalStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", paddedTerminalStoreSection));
        Check("maxwell terminal store after NOP padding",
            ((paddedTerminalStoreControl >> 8) & 0x7) == 7 &&
            (paddedTerminalStoreControl & 0xF) == 1);
        var maxwellPairSection = new rtxdisam.SassSection();
        maxwellPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        maxwellPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R5" },
        });
        ulong maxwellPairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellPairSection));
        Check("maxwell ALU/store dual issue",
            (maxwellPairControl & 0xF) == 0);
        ulong pascalPair60Control = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_60", maxwellPairSection), 0);
        ulong pascalPair62Control = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_62", maxwellPairSection), 0);
        Check("Pascal ALU/store pair parity",
            pascalPair60Control == maxwellPairControl &&
            pascalPair62Control == maxwellPairControl);
        var maxwellLeadingIaddSection = new rtxdisam.SassSection();
        maxwellLeadingIaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        maxwellLeadingIaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R5", "R6" },
        });
        maxwellLeadingIaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R7]", "R8" },
        });
        maxwellLeadingIaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong maxwellLeadingIaddControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellLeadingIaddSection));
        Check("maxwell leading IADD before slot 1-2 pair",
            (maxwellLeadingIaddControl & 0xF) == 2 &&
            ((maxwellLeadingIaddControl >> 21) & 0xF) == 0);
        var maxwellTrailingStoreSection = new rtxdisam.SassSection();
        maxwellTrailingStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        maxwellTrailingStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R5" },
        });
        maxwellTrailingStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R6]", "R7" },
        });
        maxwellTrailingStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong maxwellTrailingStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellTrailingStoreSection));
        Check("maxwell trailing store after pair",
            (maxwellTrailingStoreControl & 0xF) == 0 &&
            ((maxwellTrailingStoreControl >> 21) & 0xF) == 2);
        var sm30ExitBranchSection = new rtxdisam.SassSection();
        sm30ExitBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm30ExitBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30ExitBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm30ExitBranchControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30ExitBranchSection), 0);
        Check("SM30 EXIT/BRA terminal sections",
            ((sm30ExitBranchControl >> 4) & 0xFF) == 0x2E &&
            ((sm30ExitBranchControl >> (4 + 8)) & 0xFF) == 0);
        var sm30PaddingSection = new rtxdisam.SassSection();
        sm30PaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm30PaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        for (int nop = 0; nop < 4; nop++)
        {
            sm30PaddingSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { "RZ", "RZ" },
            });
        }
        sm30PaddingSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30PaddingSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm30PaddingControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30PaddingSection), 0);
        Check("SM30 padding MOV template",
            ((sm30PaddingControl >> 4) & 0xFF) == 0x30 &&
            ((sm30PaddingControl >> (4 + 8)) & 0xFF) == 0x2F &&
            ((sm30PaddingControl >> (4 + 2 * 8)) & 0xFF) == 0);
        var sm30MovPairSection = new rtxdisam.SassSection();
        sm30MovPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm30MovPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        sm30MovPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30MovPairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30MovPairSection), 0);
        Check("SM30 MOV/MOV/EXIT pairing",
            ((sm30MovPairControl >> 4) & 0xFF) == 0x04 &&
            ((sm30MovPairControl >> (4 + 8)) & 0xFF) == 0x2F);
        var sm30LoadStoreSection = new rtxdisam.SassSection();
        sm30LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R1", "[R2]" },
        });
        sm30LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R1" },
        });
        sm30LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30LoadStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30LoadStoreSection), 0);
        Check("SM30 load/store template",
            ((sm30LoadStoreControl >> 4) & 0xFF) == 0x30 &&
            ((sm30LoadStoreControl >> (4 + 8)) & 0xFF) == 0x2F);
        var sm30MovStoreSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R4", "R3"), ("R2", "R5"), ("R3", "R6"),
        })
        {
            sm30MovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm30MovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R4" },
        });
        sm30MovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm30MovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30MovStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm30MovStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30MovStoreSection), 0);
        Check("SM30 MOV-fed store/EXIT sections",
            ((sm30MovStoreControl >> 4) & 0xFF) == 0x30 &&
            ((sm30MovStoreControl >> (4 + 8)) & 0xFF) == 0x04 &&
            ((sm30MovStoreControl >> (4 + 2 * 8)) & 0xFF) == 0x20 &&
            ((sm30MovStoreControl >> (4 + 3 * 8)) & 0xFF) == 0x28 &&
            ((sm30MovStoreControl >> (4 + 4 * 8)) & 0xFF) == 0x04 &&
            ((sm30MovStoreControl >> (4 + 5 * 8)) & 0xFF) == 0x2E);
        var sm30AluStoreSection = new rtxdisam.SassSection();
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "R3" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "R5" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R4", "R6" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R4" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30AluStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm30AluStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30AluStoreSection), 0);
        Check("SM30 ALU-fed store/EXIT sections",
            ((sm30AluStoreControl >> (4 + 4 * 8)) & 0xFF) == 0x04 &&
            ((sm30AluStoreControl >> (4 + 5 * 8)) & 0xFF) == 0x2F);
        var sm30SharedSection = new rtxdisam.SassSection();
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R4", "[R4]" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R0", "R0", "0x2" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[R0]", "R4" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BAR.SYNC",
            Operands = { "0x0" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R6", "[0x8]" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R6" },
        });
        sm30SharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30SharedControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30SharedSection), 0);
        Check("SM30 shared-memory bundle sections",
            ((sm30SharedControl >> 4) & 0xFF) == 0x04 &&
            ((sm30SharedControl >> (4 + 8)) & 0xFF) == 0x3F &&
            ((sm30SharedControl >> (4 + 2 * 8)) & 0xFF) == 0x2E &&
            ((sm30SharedControl >> (4 + 3 * 8)) & 0xFF) == 0x2E &&
            ((sm30SharedControl >> (4 + 4 * 8)) & 0xFF) == 0x37 &&
            ((sm30SharedControl >> (4 + 5 * 8)) & 0xFF) == 0x04 &&
            ((sm30SharedControl >> (4 + 6 * 8)) & 0xFF) == 0x2F);
        var sm30BranchOnlySection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"),
            ("R7", "R8"), ("R9", "R10"), ("R11", "R12"),
        })
        {
            sm30BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm30BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm30BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm30BranchOnlySection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] sm30BranchOnlyBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_30", sm30BranchOnlySection);
        ulong sm30BranchOnlyWord = BinaryPrimitives.ReadUInt64LittleEndian(
            sm30BranchOnlyBytes.AsSpan(0x40));
        Check("SM30 branch-only bundle",
            PreVoltaDecoder.Decode(sm30BranchOnlyWord, 0x40).Text
                .StartsWith("BRA") &&
            BinaryPrimitives.ReadUInt64LittleEndian(
                sm30BranchOnlyBytes.AsSpan(0x48)) == 0x4000000000001DE4UL);
        var sm35BranchOnlySection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"),
            ("R7", "R8"), ("R9", "R10"), ("R11", "R12"),
        })
        {
            sm35BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        for (int nop = 0; nop < 7; nop++)
        {
            sm35BranchOnlySection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "NOP",
            });
        }
        sm35BranchOnlySection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] sm35BranchOnlyBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", sm35BranchOnlySection);
        PreVoltaDecoder.UseArchitecture("sm_35");
        ulong sm35BranchOnlyWord = BinaryPrimitives.ReadUInt64LittleEndian(
            sm35BranchOnlyBytes.AsSpan(0x40));
        Check("SM35 branch-only trailing NOPs",
            sm35BranchOnlyBytes.Length == 128 &&
            PreVoltaDecoder.Decode(sm35BranchOnlyWord, 0x40).Text
                .StartsWith("BRA") &&
            BinaryPrimitives.ReadUInt64LittleEndian(
                sm35BranchOnlyBytes.AsSpan(0x48)) == 0x85800000001C3C02UL);
        var sm35BrxLayoutSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8"),
        })
        {
            sm35BrxLayoutSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35BrxLayoutSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRX",
            Operands = { "R9", "-0x30" },
        });
        sm35BrxLayoutSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "RZ", "RZ" },
        });
        sm35BrxLayoutSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] sm35BrxLayoutBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", sm35BrxLayoutSection);
        PreVoltaDecoder.UseArchitecture("sm_35");
        var sm35BrxLayoutDecoded = PreVoltaDecoder
            .DecodeSection(sm35BrxLayoutBytes)
            .ToList();
        Check("SM35 BRX keeps non-final slot",
            sm35BrxLayoutDecoded.Any(item =>
                item.Instruction.Mnemonic == "BRX" &&
                item.Address == 0x28));
        var sm30LoadAddSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"),
        })
        {
            sm30LoadAddSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm30LoadAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R7", "[R8]" },
        });
        sm30LoadAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R9", "R7", "R10" },
        });
        sm30LoadAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R11]", "R9" },
        });
        sm30LoadAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm30LoadAddControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sm30LoadAddSection), 0);
        Check("SM30 load/add/store section",
            ((sm30LoadAddControl >> (4 + 3 * 8)) & 0xFF) == 0x3F &&
            ((sm30LoadAddControl >> (4 + 4 * 8)) & 0xFF) == 0x28 &&
            ((sm30LoadAddControl >> (4 + 5 * 8)) & 0xFF) == 0x04 &&
            ((sm30LoadAddControl >> (4 + 6 * 8)) & 0xFF) == 0x2F);
        var sm35PairSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8"),
        })
        {
            sm35PairSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        foreach (var (destination, left, right) in new[]
        {
            ("R9", "R10", "R11"),
            ("R12", "R13", "R14"),
            ("R15", "R16", "R17"),
        })
        {
            sm35PairSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "IADD",
                Operands = { destination, left, right },
            });
        }
        ulong sm35PairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35PairSection), 0);
        Check("SM35 full-bundle pairing",
            ((sm35PairControl >> (2 + 8)) & 0xFF) == 0x04 &&
            ((sm35PairControl >> (2 + 2 * 8)) & 0xFF) == 0x28 &&
            ((sm35PairControl >> (2 + 3 * 8)) & 0xFF) == 0x04 &&
            ((sm35PairControl >> (2 + 4 * 8)) & 0xFF) == 0x28);
        ulong sm37PairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_37", sm35PairSection), 0);
        Check("SM37 section parity", sm37PairControl == sm35PairControl);
        var sm35SelfMoveSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R1"), ("R2", "R2"), ("R3", "R3"),
            ("R4", "R4"), ("R5", "R5"), ("R6", "R6"),
        })
        {
            sm35SelfMoveSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35SelfMoveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm35SelfMoveControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35SelfMoveSection), 0);
        Check("SM35 self-move bundle",
            ((sm35SelfMoveControl >> 2) & 0xFF) == 0x30 &&
            ((sm35SelfMoveControl >> (2 + 8)) & 0xFF) == 0x24 &&
            ((sm35SelfMoveControl >> (2 + 3 * 8)) & 0xFF) == 0x3F);
        var sm35SparseExitSection = new rtxdisam.SassSection();
        sm35SparseExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm35SparseExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "RZ", "RZ" },
        });
        sm35SparseExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35SparseExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35SparseExitSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35SparseExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35SparseExitSection), 0);
        Check("SM35 sparse EXIT pairing",
            ((sm35SparseExitControl >> 2) & 0xFF) == 0x30 &&
            ((sm35SparseExitControl >> (2 + 8)) & 0xFF) == 0x04 &&
            ((sm35SparseExitControl >> (2 + 2 * 8)) & 0xFF) == 0x2F &&
            ((sm35SparseExitControl >> (2 + 3 * 8)) & 0xFF) == 0);
        var sm35UniformBranchSection = new rtxdisam.SassSection();
        sm35UniformBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA.U",
            Operands = { "`.L1`" },
        });
        sm35UniformBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35UniformBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35UniformBranchControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35UniformBranchSection), 0);
        Check("SM35 uniform branch EXIT",
            ((sm35UniformBranchControl >> (2 + 8)) & 0xFF) == 0x25);
        var sm35IntegerExitSection = new rtxdisam.SassSection();
        sm35IntegerExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        sm35IntegerExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm35IntegerExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35IntegerExitSection), 0);
        Check("SM35 integer EXIT slot",
            ((sm35IntegerExitControl >> (2 + 8)) & 0xFF) == 0x04);
        var sm35ConvertExitSection = new rtxdisam.SassSection();
        sm35ConvertExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F32.S32",
            Operands = { "R1", "R2" },
        });
        sm35ConvertExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.S32.F32.TRUNC",
            Operands = { "R3", "R1" },
        });
        sm35ConvertExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm35ConvertExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35ConvertExitSection), 0);
        Check("SM35 convert EXIT slot",
            ((sm35ConvertExitControl >> (2 + 8)) & 0xFF) == 0x04);
        var sm35ControlExitSection = new rtxdisam.SassSection();
        foreach (var controlOp in new[] { "SSY", "PBK", "PCNT" })
        {
            sm35ControlExitSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = controlOp,
                Operands = { "`.L1`" },
            });
        }
        sm35ControlExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35ControlExitSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35ControlExitControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35ControlExitSection), 0);
        Check("SM35 control EXIT slots",
            ((sm35ControlExitControl >> 2) & 0xFF) == 0x30 &&
            ((sm35ControlExitControl >> (2 + 8)) & 0xFF) == 0x24);
        var sm35BrxSection = new rtxdisam.SassSection();
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R3" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L",
            Operands = { "R0", "RZ", "0x2", "R0" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC",
            Operands = { "R2", "c[0x2][R0]" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRX",
            Operands = { "R2", "-0x30" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "RZ", "RZ" },
        });
        sm35BrxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm35BrxControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35BrxSection), 0);
        Check("SM35 BRX branch bundle sections",
            ((sm35BrxControl >> 2) & 0xFF) == 0x30 &&
            ((sm35BrxControl >> (2 + 8)) & 0xFF) == 0x28 &&
            ((sm35BrxControl >> (2 + 3 * 8)) & 0xFF) == 0x28 &&
            ((sm35BrxControl >> (2 + 4 * 8)) & 0xFF) == 0x2E &&
            ((sm35BrxControl >> (2 + 5 * 8)) & 0xFF) == 0x04 &&
            ((sm35BrxControl >> (2 + 6 * 8)) & 0xFF) == 0x2F);
        var sm35BranchPairSection = new rtxdisam.SassSection();
        for (int pair = 0; pair < 2; pair++)
        {
            sm35BranchPairSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { "RZ", "RZ" },
            });
            sm35BranchPairSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "EXIT",
            });
        }
        sm35BranchPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35BranchPairSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35BranchPairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35BranchPairSection), 0);
        Check("SM35 branch-only MOV/EXIT pairs",
            ((sm35BranchPairControl >> 2) & 0xFF) == 0x04 &&
            ((sm35BranchPairControl >> (2 + 8)) & 0xFF) == 0x2F &&
            ((sm35BranchPairControl >> (2 + 2 * 8)) & 0xFF) == 0x04 &&
            ((sm35BranchPairControl >> (2 + 3 * 8)) & 0xFF) == 0x2F &&
            ((sm35BranchPairControl >> (2 + 4 * 8)) & 0xFF) == 0);
        var sm35WideSection = new rtxdisam.SassSection();
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "c[0x0][0x44]" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R0", "SR_TID.X" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R3", "0x10" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISCADD",
            Operands = { "R4.CC", "R0", "c[0x0][0x148]", "0x4" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.U32.U32.HI.X",
            Operands = { "R5", "R0", "R3", "c[0x0][0x14c]" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E.128",
            Operands = { "R4", "[R4+0x10]" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISCADD",
            Operands = { "R2.CC", "R0", "c[0x0][0x140]", "0x4" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.U32.U32.HI.X",
            Operands = { "R3", "R0", "R3", "c[0x0][0x144]" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E.128",
            Operands = { "[R2]", "R4" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35WideSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        byte[] sm35WideBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", sm35WideSection);
        ulong sm35WideLoadControl = BinaryPrimitives.ReadUInt64LittleEndian(
            sm35WideBytes.AsSpan(0));
        ulong sm35WideStoreControl = BinaryPrimitives.ReadUInt64LittleEndian(
            sm35WideBytes.AsSpan(0x40));
        Check("SM35 wide load/store sections",
            ((sm35WideLoadControl >> 2) & 0xFF) == 0x30 &&
            ((sm35WideLoadControl >> (2 + 3 * 8)) & 0xFF) == 0x2C &&
            ((sm35WideLoadControl >> (2 + 5 * 8)) & 0xFF) == 0x04 &&
            ((sm35WideLoadControl >> (2 + 6 * 8)) & 0xFF) == 0x2C &&
            ((sm35WideStoreControl >> 2) & 0xFF) == 0x3F &&
            ((sm35WideStoreControl >> (2 + 8)) & 0xFF) == 0x04 &&
            ((sm35WideStoreControl >> (2 + 2 * 8)) & 0xFF) == 0x2F);
        var sm35AluFullSection = new rtxdisam.SassSection();
        for (int i = 1; i <= 6; i++)
        {
            sm35AluFullSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "IADD",
                Operands = { "R0", "R0", $"R{i}" },
            });
        }
        sm35AluFullSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GT.AND",
            Operands = { "P0", "PT", "R0", "0xa", "PT" },
        });
        ulong sm35AluFullControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35AluFullSection), 0);
        Check("SM35 full ALU sections",
            ((sm35AluFullControl >> 2) & 0xFF) == 0x28 &&
            ((sm35AluFullControl >> (2 + 5 * 8)) & 0xFF) == 0x28 &&
            ((sm35AluFullControl >> (2 + 6 * 8)) & 0xFF) == 0x20);
        var sm35AluStoreSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R4", "R3"), ("R2", "R5"),
        })
        {
            sm35AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R4", "R6" },
        });
        sm35AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R2]", "R4" },
        });
        sm35AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35AluStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35AluStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35AluStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35AluStoreSection), 0);
        Check("SM35 ALU-fed store/EXIT sections",
            ((sm35AluStoreControl >> (2 + 4 * 8)) & 0xFF) == 0x04 &&
            ((sm35AluStoreControl >> (2 + 5 * 8)) & 0xFF) == 0x2F);
        ulong sm32PairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_32", sm35PairSection), 0);
        Check("SM32 section parity", sm32PairControl == sm35PairControl);
        var sm35ExitBranchSection = new rtxdisam.SassSection();
        sm35ExitBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35ExitBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35ExitBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35ExitBranchControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35ExitBranchSection), 0);
        Check("SM35 EXIT/BRA terminal sections",
            ((sm35ExitBranchControl >> 2) & 0xFF) == 0x2E &&
            ((sm35ExitBranchControl >> (2 + 8)) & 0xFF) == 0 &&
            ((sm35ExitBranchControl >> (2 + 2 * 8)) & 0xFF) == 0);
        var sm35LoadDelaySection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"),
        })
        {
            sm35LoadDelaySection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35LoadDelaySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R7", "[R8]" },
        });
        sm35LoadDelaySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R9", "[R10]" },
        });
        sm35LoadDelaySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R11", "R7", "R9" },
        });
        sm35LoadDelaySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        ulong sm35LoadDelayControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35LoadDelaySection), 0);
        Check("SM35 independent load delay",
            ((sm35LoadDelayControl >> (2 + 3 * 8)) & 0xFF) == 0x23 &&
            ((sm35LoadDelayControl >> (2 + 4 * 8)) & 0xFF) == 0x3F);
        var sm35LoadStoreSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"),
        })
        {
            sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LD.E",
            Operands = { "R7", "[R8]" },
        });
        sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.E",
            Operands = { "[R9]", "R7" },
        });
        sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`.L1`" },
        });
        sm35LoadStoreSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L1" });
        ulong sm35LoadStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sm35LoadStoreSection), 0);
        Check("SM35 load/store bundle",
            ((sm35LoadStoreControl >> (2 + 8)) & 0xFF) == 0x04 &&
            ((sm35LoadStoreControl >> (2 + 2 * 8)) & 0xFF) == 0x28 &&
            ((sm35LoadStoreControl >> (2 + 3 * 8)) & 0xFF) == 0x3F &&
            ((sm35LoadStoreControl >> (2 + 4 * 8)) & 0xFF) == 0x04 &&
            ((sm35LoadStoreControl >> (2 + 5 * 8)) & 0xFF) == 0x2F &&
            ((sm35LoadStoreControl >> (2 + 6 * 8)) & 0xFF) == 0);
        string legacyFaddPath = Path.Combine(
            Path.GetTempPath(), "rtxdisam-legacy-fadd.sass");
        string legacyFaddCubinPath = Path.ChangeExtension(
            legacyFaddPath, ".cubin");
        File.WriteAllText(legacyFaddPath,
            ".target sm_50\n" +
            ".section .text\n" +
            "FADD32I R0, R1, 1.5, .NEG.ABS;\n" +
            "EXIT;\n");
        PreVoltaAssembler.Assemble(
            legacyFaddPath, "sm_50", legacyFaddCubinPath);
        CubinFile legacyFaddCubin = CubinFile.Load(legacyFaddCubinPath);
        PreVoltaDecoder.UseArchitecture("sm_50");
        var legacyFaddDecoded = PreVoltaDecoder
            .DecodeSection(legacyFaddCubin.Sections[0].Data)
            .ToList();
        Equal("legacy FADD32I NEG.ABS",
            "FADD32I R0, R1, 1.5.NEG.ABS",
            legacyFaddDecoded[0].Instruction.Text);
        string legacyFaddAttachedPath = Path.Combine(
            Path.GetTempPath(), "rtxdisam-legacy-fadd-attached.sass");
        string legacyFaddAttachedCubinPath = Path.ChangeExtension(
            legacyFaddAttachedPath, ".cubin");
        File.WriteAllText(legacyFaddAttachedPath,
            ".target sm_35\n" +
            ".section .text\n" +
            "FADD32I R122.CC, -|R93|, -117478680.NEG.ABS;\n" +
            "EXIT;\n");
        PreVoltaAssembler.Assemble(
            legacyFaddAttachedPath, "sm_35", legacyFaddAttachedCubinPath);
        CubinFile legacyFaddAttachedCubin = CubinFile.Load(
            legacyFaddAttachedCubinPath);
        PreVoltaDecoder.UseArchitecture("sm_35");
        var legacyFaddAttachedDecoded = PreVoltaDecoder
            .DecodeSection(legacyFaddAttachedCubin.Sections[0].Data)
            .ToList();
        Equal("legacy FADD32I attached modifier",
            "FADD32I R122.CC, -|R93|, -117478680.NEG.ABS",
            legacyFaddAttachedDecoded[0].Instruction.Text);
        var maxwellMovStoreSection = new rtxdisam.SassSection();
        maxwellMovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        maxwellMovStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R3]", "R4" },
        });
        ulong maxwellMovStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellMovStoreSection));
        Check("maxwell MOV/store single issue",
            (maxwellMovStoreControl & 0xF) != 0);
        var maxwellAluPairSection = new rtxdisam.SassSection();
        maxwellAluPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        maxwellAluPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4", "R5", "R6" },
        });
        ulong maxwellAluPairControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellAluPairSection));
        Check("maxwell ALU/ALU single issue",
            (maxwellAluPairControl & 0xF) != 0);
        var maxwellDependentStoreSection = new rtxdisam.SassSection();
        maxwellDependentStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R2", "R3" },
        });
        maxwellDependentStoreSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R1" },
        });
        ulong maxwellDependentStoreControl = BitConverter.ToUInt64(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellDependentStoreSection));
        Check("maxwell dependent store single issue",
            (maxwellDependentStoreControl & 0xF) == 2);
        var maxwellMovChainSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8"),
        })
        {
            maxwellMovChainSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        maxwellMovChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] maxwellMovChainBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", maxwellMovChainSection);
        ulong maxwellMovChain0 = BitConverter.ToUInt64(
            maxwellMovChainBytes, 0);
        ulong maxwellMovChain1 = BitConverter.ToUInt64(
            maxwellMovChainBytes, 32);
        Check("maxwell MOV/MOV stalls",
            ((maxwellMovChain0 >> 42) & 0xF) == 1 &&
            (maxwellMovChain1 & 0xF) == 1);
        var maxwellMovStoreChainSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8"),
        })
        {
            maxwellMovStoreChainSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        maxwellMovStoreChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R9]", "R10" },
        });
        maxwellMovStoreChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] maxwellMovStoreChainBytes =
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_50", maxwellMovStoreChainSection);
        ulong maxwellMovStoreChain1 = BitConverter.ToUInt64(
            maxwellMovStoreChainBytes, 32);
        Check("maxwell MOV/store stall",
            (maxwellMovStoreChain1 & 0xF) == 2);
        var maxwellMovAluSection = new rtxdisam.SassSection();
        foreach (var (destination, source) in new[]
        {
            ("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8"),
        })
        {
            maxwellMovAluSection.Nodes.Add(new rtxdisam.SassInstruction
            {
                Mnemonic = "MOV",
                Operands = { destination, source },
            });
        }
        maxwellMovAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R9", "R10", "R11" },
        });
        maxwellMovAluSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] maxwellMovAluBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", maxwellMovAluSection);
        ulong maxwellMovAlu1 = BitConverter.ToUInt64(maxwellMovAluBytes, 32);
        Check("maxwell MOV/ALU stall",
            (maxwellMovAlu1 & 0xF) == 4);
        Check("SM30 default control word",
            rtxdisam.PreVoltaAssembler.DefaultControlWord("sm_30") ==
                0x200000000002f307UL);
        Check("SM50 default control word",
            rtxdisam.PreVoltaAssembler.DefaultControlWord("sm_50") ==
                0x001fc800fe2007f6UL);
        Check("pre-volta IADD timing",
            rtxdisam.PreVoltaScheduling.TryGet("IADD", out var iaddTiming) &&
            iaddTiming.Latency == 6 && iaddTiming.Throughput == 1);
        Check("pre-volta ISETP timing",
            rtxdisam.PreVoltaScheduling.TryGet("ISETP.GT.AND", out var isetpTiming) &&
            isetpTiming.Latency == 13 && isetpTiming.Throughput == 2);
        Check("pre-volta branch minimum stall",
            rtxdisam.PreVoltaScheduling.MinimumStall("BRA") == 5 &&
            rtxdisam.PreVoltaScheduling.MinimumStall("EXIT") == 5 &&
            rtxdisam.PreVoltaScheduling.MinimumStall("BRK") == 5 &&
            rtxdisam.PreVoltaScheduling.MinimumStall("SSY") == 5 &&
            rtxdisam.PreVoltaScheduling.MinimumStall("IADD") == 0);
        Check("kepler dual-issue independence",
            rtxdisam.PreVoltaScheduling.CanDualIssue(
                "IADD", new[] { "R1", "R2", "R3" },
                "MOV", new[] { "R4", "R5" }));
        Check("kepler dual-issue dependency",
            !rtxdisam.PreVoltaScheduling.CanDualIssue(
                "IADD", new[] { "R1", "R2", "R3" },
                "IADD", new[] { "R4", "R1", "R5" }));
        Check("kepler dual-issue control",
            !rtxdisam.PreVoltaScheduling.CanDualIssue(
                "MOV", new[] { "R1", "R2" },
                "EXIT", Array.Empty<string>()));
        Check("kepler dual-issue set conflict",
            !rtxdisam.PreVoltaScheduling.CanDualIssue(
                "ISETP.GT.AND", new[] { "P0", "PT", "R1", "R2", "PT" },
                "ISETP.LT.AND", new[] { "P1", "PT", "R3", "R4", "PT" }));
        Check("kepler dual-issue multiply conflict",
            !rtxdisam.PreVoltaScheduling.CanDualIssue(
                "IMUL", new[] { "R1", "R2", "R3" },
                "SHL", new[] { "R4", "R5", "R6" }));
        ulong keplerDual = rtxdisam.PreVoltaScheduling.SetKeplerDualIssue(
            0, 0, true);
        Check("kepler dual-issue bit",
            (keplerDual & (1UL << 7)) == 0);
        ulong gk104Dual = rtxdisam.PreVoltaScheduling.SetGk104DualIssue(
            0, 0, true);
        Check("GK104 dual-issue byte",
            ((gk104Dual >> 4) & 0xFF) == 0x04);
        var gk104Section = new rtxdisam.SassSection();
        gk104Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        gk104Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        gk104Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "R6" },
        });
        gk104Section.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        ulong gk104DualControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection("sm_30", gk104Section));
        Check("GK104 generated dual-issue byte",
            ((gk104DualControl >> (4 + 8)) & 0xFF) == 0x04);
        Check("GK104 store/EXIT dual issue",
            rtxdisam.PreVoltaScheduling.CanDualIssue(
                "ST", new[] { "[R1]", "R2" }, "EXIT", Array.Empty<string>(),
                kepler: false));
        ulong keplerBarrier = rtxdisam.PreVoltaScheduling.AddKeplerBarriers(
            0, 1, shared: true, global: false, texture: true);
        Check("kepler barrier bits",
            (keplerBarrier & (1UL << (2 + 8 + 4))) != 0 &&
            (keplerBarrier & (1UL << (2 + 8 + 7))) != 0);
        string explicitControlAsm = Path.GetTempFileName();
        File.WriteAllText(explicitControlAsm,
            ".target sm_35\n" +
            ".section .text.k,\"ax\",@progbits\n" +
            "k:\n" +
            "T:-:D:S:05 MOV R1, R2;\n" +
            "EXIT;\n");
        var explicitProgram = rtxdisam.SassParser.Parse(explicitControlAsm);
        var explicitSection = explicitProgram.Sections
            .First(section => section.Name == ".text.k");
        var explicitInstruction = explicitSection.Nodes
            .OfType<rtxdisam.SassInstruction>().First();
        Check("explicit control byte parse",
            explicitInstruction.ControlByte == 0x95,
            explicitInstruction.ControlByte?.ToString("x"));
        ulong explicitControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", explicitSection));
        Check("explicit control byte preserved",
            ((explicitControl >> 2) & 0xFF) == 0x95,
            ((explicitControl >> 2) & 0xFF).ToString("x"));
        Check("explicit control zeros unused slots",
            ((explicitControl >> (2 + 2 * 8)) & 0xFF) == 0);
        File.Delete(explicitControlAsm);
        string wordControlAsm = Path.GetTempFileName();
        File.WriteAllText(wordControlAsm,
            ".target sm_50\n" +
            ".section .text.k,\"ax\",@progbits\n" +
            "k:\n" +
            "W:0x001c4400fe0007f6 MOV R1, R2;\n" +
            "EXIT;\n");
        var wordProgram = rtxdisam.SassParser.Parse(wordControlAsm);
        var wordSection = wordProgram.Sections
            .First(section => section.Name == ".text.k");
        var wordInstruction = wordSection.Nodes
            .OfType<rtxdisam.SassInstruction>().First();
        Check("explicit control word parse",
            wordInstruction.ControlWord == 0x001c4400fe0007f6UL,
            wordInstruction.ControlWord?.ToString("x"));
        ulong wordControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection("sm_50", wordSection));
        Check("explicit control word preserved",
            wordControl == 0x001c4400fe0007f6UL,
            wordControl.ToString("x"));
        File.Delete(wordControlAsm);
        string instWordAsm = Path.GetTempFileName();
        File.WriteAllText(instWordAsm,
            ".target sm_50\n" +
            ".section .text.k,\"ax\",@progbits\n" +
            "k:\n" +
            "W:0x001c4400fe0007f6 I:0xe30000000007000f EXIT;\n");
        var instWordProgram = rtxdisam.SassParser.Parse(instWordAsm);
        var instWordSection = instWordProgram.Sections
            .First(section => section.Name == ".text.k");
        var instWordInstruction = instWordSection.Nodes
            .OfType<rtxdisam.SassInstruction>().First();
        Check("explicit instruction word parse",
            instWordInstruction.InstructionWord == 0xe30000000007000fUL,
            instWordInstruction.InstructionWord?.ToString("x"));
        byte[] instWordBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", instWordSection);
        Check("explicit instruction word preserved",
            BinaryPrimitives.ReadUInt64LittleEndian(
                instWordBytes.AsSpan(8)) == 0xe30000000007000fUL);
        File.Delete(instWordAsm);
        string nvHexAsm = Path.GetTempFileName();
        File.WriteAllText(nvHexAsm,
            ".target sm_50\n" +
            ".section .text.k,\"ax\",@progbits\n" +
            "k:\n" +
            "/* 0x001c4400fe0007f6 */\n" +
            "/*0008*/ EXIT; /* 0xe30000000007000f */\n");
        var nvHexProgram = rtxdisam.SassParser.Parse(nvHexAsm);
        var nvHexSection = nvHexProgram.Sections
            .First(section => section.Name == ".text.k");
        var nvHexInstruction = nvHexSection.Nodes
            .OfType<rtxdisam.SassInstruction>().First();
        Check("nvdisasm hex control word parse",
            nvHexInstruction.ControlWord == 0x001c4400fe0007f6UL,
            nvHexInstruction.ControlWord?.ToString("x"));
        Check("nvdisasm hex instruction word parse",
            nvHexInstruction.InstructionWord == 0xe30000000007000fUL,
            nvHexInstruction.InstructionWord?.ToString("x"));
        byte[] nvHexBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_50", nvHexSection);
        Check("nvdisasm hex words preserved",
            BinaryPrimitives.ReadUInt64LittleEndian(nvHexBytes.AsSpan(0)) ==
                0x001c4400fe0007f6UL &&
            BinaryPrimitives.ReadUInt64LittleEndian(nvHexBytes.AsSpan(8)) ==
                0xe30000000007000fUL);
        File.Delete(nvHexAsm);
        var liftSection = new rtxdisam.SassSection();
        liftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        liftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "c[0x0][0x144]" },
        });
        liftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        liftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x1" },
        });
        liftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        liftSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string liftedPtx = rtxdisam.SassToPtx.Translate(
            liftSection, "probe",
            new[] { new rtxdisam.PtxParameter("p", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX parameter and memory",
            liftedPtx.Contains("ld.param.u64 %rdp0, [p];") &&
            liftedPtx.Contains("ld.global.u32 %r0, [%rpptr0+0];") &&
            liftedPtx.Contains("add.s32 %r0, %r0, 0x1;") &&
            liftedPtx.Contains("st.global.u32 [%rpptr0+0], %r0;") &&
            liftedPtx.Contains("ret;"));
        var constantLiftSection = new rtxdisam.SassSection();
        constantLiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "c[0x0][0x20]" },
        });
        string constantLiftedPtx = rtxdisam.SassToPtx.Translate(
            constantLiftSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", offset => offset == 0x20 ? 42u : null);
        Check("SASS to PTX constant bank",
            constantLiftedPtx.Contains("mov.b32 %r1, 42;"));
        var fastDivSection = new rtxdisam.SassSection();
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.U32.RP",
            Operands = { "R4", "R13" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MUFU.RCP",
            Operands = { "R4", "R4" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R27", "R4", "0xffffffe", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.FTZ.U32.TRUNC.NTZ",
            Operands = { "R27", "R27" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R5", "R5", "R27", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.HI.U32",
            Operands = { "R26", "R27", "R5", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.HI.U32",
            Operands = { "R2", "R26", "R6", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.MOV",
            Operands = { "R23", "RZ", "RZ", "-R2" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R22", "R13", "R23", "R6" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GT.U32.AND",
            Operands = { "P1", "PT", "R13", "R22", "PT" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.IADD",
            Guard = "@!P1",
            Operands = { "R22", "R22", "0x1", "-R13" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Guard = "@!P1",
            Operands = { "R2", "R2", "0x1", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GE.U32.AND",
            Operands = { "P0", "PT", "R22", "R13", "PT" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Guard = "@P0",
            Operands = { "R2", "R2", "0x1", "RZ" },
        });
        fastDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R30", "R23", "0x1", "RZ" },
        });
        string fastDivPtx = rtxdisam.SassToPtx.Translate(
            fastDivSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX GT-form fast division",
            fastDivPtx.Contains("div.u32 %r2, %r6, %r13;") &&
            fastDivPtx.Contains("neg.s32"));
        var wideLeaSection = new rtxdisam.SassSection();
        wideLeaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA",
            Operands = { "R20", "P1", "R22", "c[0x0][0x160]", "0x2" },
        });
        wideLeaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R27", "R23", "0x1", "R25" },
        });
        wideLeaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA.HI.X",
            Operands = { "R27", "R22", "c[0x0][0x164]", "R27", "0x2", "P1" },
        });
        string wideLeaPtx = rtxdisam.SassToPtx.Translate(
            wideLeaSection, "k",
            new[] { new rtxdisam.PtxParameter("p", "u64", 0x160, 8) },
            "sm_80");
        int addendWrite = wideLeaPtx.IndexOf("add.s32 %r27");
        int highAdd = wideLeaPtx.IndexOf("add.u32 %r27,");
        Check("SASS to PTX LEA.HI.X live high addend",
            wideLeaPtx.Contains("add.cc.u32 %r20") &&
            wideLeaPtx.Contains("selp.b32 %rt6, 1, 0, %p1;") &&
            addendWrite >= 0 && highAdd > addendWrite);
        // ptxas can reuse the register that held the low word of a 64-bit
        // value for the high word of an addend between IADD3 and IMAD.X.
        // Folding the pair at the IADD3 captured the pre-overwrite value;
        // the argmax/max reduce kernel's translate_idx then wrote {idx,idx}.
        var clobberedPairSection = new rtxdisam.SassSection();
        clobberedPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R16", "P1", "R2", "R7", "RZ" },
        });
        clobberedPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "RZ" },
        });
        clobberedPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.X",
            Operands = { "R17", "R3", "0x1", "R2", "P1" },
        });
        // Uniform FLO.U32 (gatherKthValue/radix select) must lift like FLO.
        var ufloSection = new rtxdisam.SassSection();
        ufloSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UFLO.U32",
            Operands = { "UR4", "UR7" },
        });
        string ufloPtx = rtxdisam.SassToPtx.Translate(
            ufloSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UFLO.U32",
            ufloPtx.Contains("clz.b32") &&
            ufloPtx.Contains("sub.u32 %ur4, 31,"));
        // A predicated vector store from RZ writes zeros; the predicated
        // split must accept the zero source (gatherMedian shared counters).
        var storeRzSection = new rtxdisam.SassSection();
        storeRzSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS.64",
            Guard = "@!P0",
            Operands = { "[R4]", "RZ" },
        });
        string storeRzPtx = rtxdisam.SassToPtx.Translate(
            storeRzSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX predicated STS.64 from RZ",
            storeRzPtx.Contains("st.shared.u32") &&
            storeRzPtx.Split("st.shared.u32").Length - 1 == 2);
        string clobberedPairPtx = rtxdisam.SassToPtx.Translate(
            clobberedPairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        int clobberWrite = clobberedPairPtx.IndexOf("mov.b32 %r2, 0;");
        int clobberHigh = clobberedPairPtx.IndexOf("mov.b32 %r17");
        Check("SASS to PTX IADD3/IMAD.X high source rewritten between halves",
            clobberedPairPtx.Contains("add.cc.u32") &&
            clobberedPairPtx.Contains("addc.u32 %rt") &&
            !clobberedPairPtx.Contains("add.u64") &&
            clobberWrite >= 0 && clobberHigh > clobberWrite);
        var wideConstantSection = new rtxdisam.SassSection();
        wideConstantSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.WIDE.U32",
            Operands = { "R20", "R20", "R21", "c[0x2][0x0]" },
        });
        string wideConstantPtx = rtxdisam.SassToPtx.Translate(
            wideConstantSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80",
            constant2Resolver: offset =>
                offset == 0 || offset == 4 ? 0xFFFFFFFFu : null,
            constant2Bytes: 8);
        Check("SASS to PTX IMAD.WIDE 64-bit constant addend",
            wideConstantPtx.Contains("{0xffffffff, 0xffffffff}") &&
            wideConstantPtx.Contains("mad.wide.u32"));
        var divHelperSection = new rtxdisam.SassSection();
        divHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Operands = { "0x100" },
            Relocations =
            {
                new rtxdisam.SassRelocation
                {
                    Symbol = "$__internal_46_$__cuda_sm20_div_s64",
                },
            },
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "0x40" },
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.U32.RP",
            Operands = { "R12", "R13" },
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = ".L_merge",
            Address = 0x40,
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.MOV.U32",
            Operands = { "R9", "RZ", "RZ", "R15" },
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        divHelperSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = "$__internal_46_$__cuda_sm20_div_s64",
            Address = 0x100,
        });
        foreach (var (mnemonic, operands) in new[]
        {
            ("IADD3", new[] { "R22", "P1", "RZ", "-R15", "RZ" }),
            ("ISETP.GE.AND", new[] { "P0", "PT", "R12", "RZ", "PT" }),
            ("ISETP.GE.AND", new[] { "P2", "PT", "R29", "RZ", "PT" }),
            ("IMAD.X", new[] { "R23", "RZ", "RZ", "~R12", "P1" }),
            ("SEL", new[] { "R22", "R22", "R15", "!P0" }),
            ("I2F.U64.RP", new[] { "R28", "R22" }),
            ("MUFU.RCP", new[] { "R28", "R28" }),
            ("IADD3", new[] { "R31", "P3", "RZ", "-R20", "RZ" }),
            ("IMAD.X", new[] { "R20", "RZ", "RZ", "~R29", "P3" }),
            ("SEL", new[] { "R15", "R15", "R20", "!P0" }),
            ("RET.REL.NODEC", new[] { "R22", "0x0" }),
        })
        {
            var instruction = new rtxdisam.SassInstruction { Mnemonic = mnemonic };
            instruction.Operands.AddRange(operands);
            divHelperSection.Nodes.Add(instruction);
        }
        string divHelperPtx = rtxdisam.SassToPtx.Translate(
            divHelperSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX per-module div_s64 ABI",
            divHelperPtx.Contains("div.s64") &&
            divHelperPtx.Contains("{%r20, %r29}") &&
            divHelperPtx.Contains("{%r15, %r12}") &&
            divHelperPtx.Contains("mov.b32 %r15,"));
        var dsqrtHelperSection = new rtxdisam.SassSection();
        dsqrtHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Operands = { "0x100" },
            Relocations =
            {
                new rtxdisam.SassRelocation
                {
                    Symbol = "$__internal_5_$__cuda_sm20_dsqrt_rn_f64_mediumpath_v1",
                },
            },
        });
        dsqrtHelperSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string dsqrtHelperPtx = rtxdisam.SassToPtx.Translate(
            dsqrtHelperSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX dsqrt_rn_f64 slowpath ABI",
            dsqrtHelperPtx.Contains("sqrt.rn.f64") &&
            dsqrtHelperPtx.Contains("{%r32, %r33}") &&
            dsqrtHelperPtx.Contains("{%r20, %r21}"));
        // An inlined helper with a dynamic stack adjustment must get
        // the __callee_dynamic_frame declaration even though the header
        // pre-scan may not see its (register-sized) frame move.
        var dynamicFrameSection = new rtxdisam.SassSection();
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Operands = { "0x100" },
            Relocations =
            {
                new rtxdisam.SassRelocation
                {
                    Symbol = "$__internal_1_$__cuda_sm20_div_rn_f64_full",
                },
            },
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = "$__internal_1_$__cuda_sm20_div_rn_f64_full",
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R1", "R5" },
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDL.64",
            Operands = { "R2", "[R1+0x8]" },
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STL.64",
            Operands = { "[R1+0x10]", "R2" },
        });
        dynamicFrameSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "RET",
        });
        string inlineDynFramePtx = rtxdisam.SassToPtx.Translate(
            dynamicFrameSection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX inlined dynamic local frame is declared",
            inlineDynFramePtx.Contains(
                ".local .align 16 .b8 __callee_dynamic_frame[65536];") &&
            inlineDynFramePtx.Contains("__callee_dynamic_frame;"));
        var frnd64Section = new rtxdisam.SassSection();
        frnd64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FRND.F64",
            Operands = { "R24", "R20" },
        });
        frnd64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string frnd64Ptx = rtxdisam.SassToPtx.Translate(
            frnd64Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX FRND.F64 keeps the register pair",
            frnd64Ptx.Contains("cvt.rni.f64.f64") &&
            frnd64Ptx.Contains("{%r24, %r25}") &&
            frnd64Ptx.Contains("{%r20, %r21}"));
        var predMul64Section = new rtxdisam.SassSection();
        predMul64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DMUL",
            Guard = "@!P2",
            Operands = { "R34", "RZ", "R18" },
        });
        predMul64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DFMA",
            Guard = "@P2",
            Operands = { "R34", "R18", "R20", "R22" },
        });
        predMul64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string predMul64Ptx = rtxdisam.SassToPtx.Translate(
            predMul64Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX predicated DMUL/DFMA guard the pair write",
            predMul64Ptx.Contains("@!%p2 mul.f64") &&
            predMul64Ptx.Contains("@!%p2 mov.b64 {%r34, %r35}") &&
            predMul64Ptx.Contains("@%p2 fma.rn.f64") &&
            predMul64Ptx.Contains("@%p2 mov.b64 {%r34, %r35}"));
        var r2pSection = new rtxdisam.SassSection();
        r2pSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "R2P",
            Operands = { "PR", "R0", "0x3" },
        });
        r2pSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string r2pPtx = rtxdisam.SassToPtx.Translate(
            r2pSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX R2P copies selected predicate bits",
            r2pPtx.Contains("setp.ne.u32 %p0") &&
            r2pPtx.Contains("setp.ne.u32 %p1") &&
            !r2pPtx.Contains("setp.ne.u32 %p2"));
        // R2P redefines P0..P6; guards after it must read the fresh
        // snapshot, not the predicate's previous definition.
        var r2pSnapshotSection = new rtxdisam.SassSection();
        r2pSnapshotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.NE.AND",
            Operands = { "P1", "PT", "R0", "RZ", "PT" },
        });
        r2pSnapshotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "R2P",
            Operands = { "PR", "R0", "0x3" },
        });
        r2pSnapshotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Guard = "@P1",
            Operands = { "R5", "R5", "0x1", "RZ" },
        });
        r2pSnapshotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string r2pSnapshotPtx = rtxdisam.SassToPtx.Translate(
            r2pSnapshotSection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        MatchCollection p1Snapshots = System.Text.RegularExpressions.Regex.Matches(
            r2pSnapshotPtx, @"mov\.pred (%ps\d+), %p1;");
        string freshP1 = p1Snapshots.Count > 0
            ? p1Snapshots[^1].Groups[1].Value
            : "";
        Check("SASS to PTX R2P refreshes predicate snapshots",
            freshP1.Length > 0 &&
            r2pSnapshotPtx.Contains("@" + freshP1 + " add.s32 %r5"));

        // The f64 Payne-Hanek slow path is a leaf whose register ABI
        // varies per kernel; the lifter recovers it and expands the
        // libdevice algorithm.  Guard the ABI derivation and the exact
        // fixed-point multiplier (0xC90FDAA22168C235, not the decimal
        // constant's apparent 0xB8EF... value).
        const string trigSymbol =
            "$__internal_7_$__internal_trig_reduction_slowpathd";
        var trigSection = new rtxdisam.SassSection();
        trigSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Operands = { "0x100" },
            Relocations =
            {
                new rtxdisam.SassRelocation { Symbol = trigSymbol },
            },
        });
        trigSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        trigSection.Nodes.Add(new rtxdisam.SassLabel { Name = trigSymbol });
        foreach (var (mnemonic, operands, guard) in new[]
        {
            ("SHF.R.U32.HI", new[] { "R13", "RZ", "0x14", "R11" }, (string?)null),
            ("IMAD.MOV.U32", new[] { "R0", "RZ", "RZ", "R10" }, null),
            // A dynamic local-frame store keeps this body out of the
            // inline lifter, exactly like the real slow path.
            ("STL.64", new[] { "[R0]", "R14" }, null),
            ("SHF.L.U64.HI", new[] { "R21", "R0", "0xb", "R11" }, null),
            ("IADD3", new[] { "R2", "P2", "P3", "R3", "-UR4", "RZ" }, null),
            ("LOP3.LUT", new[] { "R11", "R2", "R11", "RZ", "0xfc", "!PT" }, null),
            ("IMAD.MOV", new[] { "R5", "RZ", "RZ", "-R5" }, "@P0"),
            ("IMAD.MOV.U32", new[] { "R2", "RZ", "RZ", "R0" }, null),
            ("IMAD.MOV.U32", new[] { "R3", "RZ", "RZ", "R11" }, null),
            ("RET.REL.NODEC", new[] { "R12", "0x0" }, null),
        })
        {
            var instruction = new rtxdisam.SassInstruction
            {
                Mnemonic = mnemonic,
                Guard = guard,
            };
            instruction.Operands.AddRange(operands);
            trigSection.Nodes.Add(instruction);
        }
        string trigPtx = rtxdisam.SassToPtx.Translate(
            trigSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX trig slowpath recovers the register ABI",
            System.Text.RegularExpressions.Regex.IsMatch(
                trigPtx, @"mov\.b64 %rdt\d+, \{%r10, %r11\};") &&
            System.Text.RegularExpressions.Regex.IsMatch(
                trigPtx, @"mov\.b64 \{%r2, %r3\}, %rdt\d+;") &&
            trigPtx.Contains("clz.b64") &&
            trigPtx.Contains("mul.hi.u64") &&
            trigPtx.Contains("0xc90fdaa22168c235") &&
            trigPtx.Contains("ld.const.u64") &&
            trigPtx.Contains("__rtx_trig_i2opi_d[18]"));

        // Bank 4 relocations name module data; bank 2 holds raw math
        // constants whose offsets collide (0x0, 0x8, ...).  Only bank 4
        // may materialize a symbol address.
        var globalImages = new Dictionary<int, PtxGlobalImage>
        {
            [0] = new PtxGlobalImage(
                "__cudart_test_table", 0,
                new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }),
        };
        var bank4Section = new rtxdisam.SassSection();
        bank4Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.WIDE.U32",
            Operands = { "R34", "R16", "R35", "c[0x4][0x0]" },
        });
        bank4Section.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string bank4Ptx = rtxdisam.SassToPtx.Translate(
            bank4Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80",
            constant4Resolver: _ => 0u, constant4Bytes: 0x100,
            globalImages: globalImages);
        Check("SASS to PTX bank-4 relocation materializes a global image",
            bank4Ptx.Contains("mov.u64") &&
            bank4Ptx.Contains("__cudart_test_table") &&
            bank4Ptx.Contains(
                ".global .align 8 .b8 __cudart_test_table[8]"));
        var bank2Section = new rtxdisam.SassSection();
        bank2Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DMUL",
            Operands = { "R20", "R16", "c[0x2][0x0]" },
        });
        bank2Section.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string bank2Ptx = rtxdisam.SassToPtx.Translate(
            bank2Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80",
            constant2Resolver: _ => 0u, constant2Bytes: 8,
            globalImages: globalImages);
        Check("SASS to PTX bank-2 constants do not alias bank-4 globals",
            !System.Text.RegularExpressions.Regex.IsMatch(
                bank2Ptx,
                @"mov\.u64 %rdt\d+, __cudart_test_table;") &&
            bank2Ptx.Contains("f64"));
        var notConstSection = new rtxdisam.SassSection();
        notConstSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R1", "RZ", "~c[0x0][0x19c]", "RZ" },
        });
        notConstSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        string notConstPtx = rtxdisam.SassToPtx.Translate(
            notConstSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80",
            constantResolver: _ => 0x1234u);
        Check("SASS to PTX complements a constant-bank operand",
            notConstPtx.Contains("not.b32") &&
            !notConstPtx.Contains("~c["));
        var xmadLiftSection = new rtxdisam.SassSection();
        xmadLiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R5", "R0.reuse", "0x5", "R5" },
        });
        xmadLiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.PSL",
            Operands = { "R0", "R0.H1", "0x5", "R5" },
        });
        string xmadLiftedPtx = rtxdisam.SassToPtx.Translate(
            xmadLiftSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX XMAD reuse and half",
            xmadLiftedPtx.Contains("mad.lo.s32 %r5, %r0, 0x5, %r5;") &&
            xmadLiftedPtx.Contains("shr.u32 %rt0, %r0, 16;") &&
            xmadLiftedPtx.Contains("mul.lo.u32 %rt1, %rt0, 0x5;") &&
            !xmadLiftedPtx.Contains(".reuse") &&
            !xmadLiftedPtx.Contains(".H1"));
        var plopSection = new rtxdisam.SassSection();
        plopSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PLOP3.LUT",
            Operands = { "P0", "PT", "PT", "PT", "PT", "0x8", "0x0" },
        });
        string plopPtx = rtxdisam.SassToPtx.Translate(
            plopSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX PLOP3 constant LUT",
            plopPtx.Contains("mov.pred %p0, 0;"));
        var plopVariableSection = new rtxdisam.SassSection();
        plopVariableSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PLOP3.LUT",
            Operands = { "P0", "P1", "P2", "P3", "PT", "0x8", "0x0" },
        });
        string plopVariablePtx = rtxdisam.SassToPtx.Translate(
            plopVariableSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_80");
        Check("SASS to PTX PLOP3 predicate LUT",
            plopVariablePtx.Contains("not.pred") &&
            plopVariablePtx.Contains("and.pred") &&
            plopVariablePtx.Contains("mov.pred %p0,"));
        // The printed leading predicate is the control operand; the LUT is
        // over the last three.  P0 = UP0 is PLOP3 ... PT,PT,PT,UP0,0x80.
        var plopMoveSection = new rtxdisam.SassSection();
        plopMoveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PLOP3.LUT",
            Operands = { "P0", "PT", "PT", "PT", "UP0", "0x80", "0x0" },
        });
        string plopMovePtx = rtxdisam.SassToPtx.Translate(
            plopMoveSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX PLOP3 predicate move",
            plopMovePtx.Contains("mov.pred %p0, %up0;"));
        var plopTrueSection = new rtxdisam.SassSection();
        plopTrueSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PLOP3.LUT",
            Operands = { "P0", "PT", "PT", "PT", "PT", "0x80", "0x0" },
        });
        string plopTruePtx = rtxdisam.SassToPtx.Translate(
            plopTrueSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX PLOP3 constant true",
            plopTruePtx.Contains("mov.pred %p0, 1;"));
        var uaddSection = new rtxdisam.SassSection();
        uaddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3",
            Operands = { "UR4", "UR4", "-0x3", "URZ" },
        });
        string uaddPtx = rtxdisam.SassToPtx.Translate(
            uaddSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIADD3 immediate",
            uaddPtx.Contains("add.u32 %ur4, %ur4, 0xfffffffd;"));
        var uaddNegSection = new rtxdisam.SassSection();
        uaddNegSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3",
            Operands = { "UR4", "UR6", "-UR8", "URZ" },
        });
        string uaddNegPtx = rtxdisam.SassToPtx.Translate(
            uaddNegSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIADD3 negated uniform register",
            uaddNegPtx.Contains("sub.u32 %rt") &&
            uaddNegPtx.Contains("add.u32 %ur4, %ur6, %rt"));
        var uaddNotSection = new rtxdisam.SassSection();
        uaddNotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3",
            Operands = { "UR4", "UR6", "~UR8", "URZ" },
        });
        string uaddNotPtx = rtxdisam.SassToPtx.Translate(
            uaddNotSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIADD3 inverted uniform register",
            uaddNotPtx.Contains("not.b32 %rt") &&
            uaddNotPtx.Contains("%ur8") &&
            uaddNotPtx.Contains("add.u32 %ur4, %ur6, %rt"));
        var uaddPairSection = new rtxdisam.SassSection();
        uaddPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3",
            Operands = { "UR6", "UP0", "UR4", "-0x1", "URZ" },
        });
        uaddPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3.X",
            Operands = { "UR7", "UR5", "-0x1", "URZ", "UP0", "!UPT" },
        });
        string uaddPairPtx = rtxdisam.SassToPtx.Translate(
            uaddPairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIADD3.X 64-bit pair",
            uaddPairPtx.Contains("add.u64") &&
            uaddPairPtx.Contains("mov.b64 {%ur6, %ur7}"));
        var uimadSection = new rtxdisam.SassSection();
        uimadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIMAD",
            Operands = { "UR4", "UR4", "UR5", "URZ" },
        });
        string uimadPtx = rtxdisam.SassToPtx.Translate(
            uimadSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIMAD",
            uimadPtx.Contains("mad.lo.u32 %ur4, %ur4, %ur5, 0;"));
        var uimadWideSection = new rtxdisam.SassSection();
        uimadWideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIMAD.WIDE.U32",
            Operands = { "UR4", "UR4", "UR5", "URZ" },
        });
        string uimadWidePtx = rtxdisam.SassToPtx.Translate(
            uimadWideSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIMAD.WIDE",
            uimadWidePtx.Contains("mul.wide.u32") &&
            uimadWidePtx.Contains("mov.b64 {%ur4, %ur5}"));
        var uleaSection = new rtxdisam.SassSection();
        uleaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ULEA",
            Operands = { "UR4", "UR2", "UR3", "0x2" },
        });
        string uleaPtx = rtxdisam.SassToPtx.Translate(
            uleaSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ULEA uniform shift-add",
            uleaPtx.Contains("mad.lo.u32 %ur4, %ur2, 4, %ur3;"));
        var ushfSection = new rtxdisam.SassSection();
        ushfSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "USHF.L.U32",
            Operands = { "UR4", "UR4", "0x2", "URZ" },
        });
        string ushfPtx = rtxdisam.SassToPtx.Translate(
            ushfSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX USHF.L.U32",
            ushfPtx.Contains("shf.l.wrap.b32 %ur4, 0, %ur4, 0x2;"));
        var ushfBlockBaseSection = new rtxdisam.SassSection();
        ushfBlockBaseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "USHF.L.U32",
            Operands = { "UR6", "UR6", "0xa", "URZ" },
        });
        string ushfBlockBasePtx = rtxdisam.SassToPtx.Translate(
            ushfBlockBaseSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX USHF.L.U32 block base",
            ushfBlockBasePtx.Contains("shf.l.wrap.b32 %ur6, 0, %ur6, 0xa;"));
        var shfLeftLowSection = new rtxdisam.SassSection();
        shfLeftLowSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L.U32",
            Operands = { "R0", "R0", "0xa", "RZ" },
        });
        string shfLeftLowPtx = rtxdisam.SassToPtx.Translate(
            shfLeftLowSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SHF.L.U32 low half",
            shfLeftLowPtx.Contains("shf.l.wrap.b32 %r0, 0, %r0, 0xa;"));
        var shfLeftHighSection = new rtxdisam.SassSection();
        shfLeftHighSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L.U32.HI",
            Operands = { "R5", "R10", "0x2", "R9" },
        });
        string shfLeftHighPtx = rtxdisam.SassToPtx.Translate(
            shfLeftHighSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SHF.L.U32.HI high half",
            shfLeftHighPtx.Contains("shf.l.wrap.b32 %r5, %r10, %r9, 0x2;"));
        // A dynamic SASS shift covers 0..63.  PTX shf clamps/wraps at
        // 32, so the lifter must use a real 64-bit shift; the radix
        // sort's high-digit passes were no-ops before this.
        var shfDynamicSection = new rtxdisam.SassSection();
        shfDynamicSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U32.HI",
            Operands = { "R0", "R1", "R5", "R2" },
        });
        string shfDynamicPtx = rtxdisam.SassToPtx.Translate(
            shfDynamicSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX dynamic SHF covers the full 64-bit count",
            shfDynamicPtx.Contains("mov.b64 %rdt") &&
            shfDynamicPtx.Contains("and.b32 %rt") &&
            shfDynamicPtx.Contains("0x3f;") &&
            shfDynamicPtx.Contains("shr.u64") &&
            !shfDynamicPtx.Contains("shf.r."));
        var shfDynamicLeftSection = new rtxdisam.SassSection();
        shfDynamicLeftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L.U32",
            Operands = { "R0", "R1", "R5", "R2" },
        });
        string shfDynamicLeftPtx = rtxdisam.SassToPtx.Translate(
            shfDynamicLeftSection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX dynamic SHF.L uses a 64-bit left shift",
            shfDynamicLeftPtx.Contains("shl.b64") &&
            !shfDynamicLeftPtx.Contains("shf.l."));
        var uldcByteSection = new rtxdisam.SassSection();
        uldcByteSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ULDC.U8",
            Operands = { "UR4", "c[0x0][0x571]" },
        });
        string uldcBytePtx = rtxdisam.SassToPtx.Translate(
            uldcByteSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ULDC.U8 narrow load",
            uldcBytePtx.Contains("shr.u32 %ur4, %ur4, 8;") &&
            uldcBytePtx.Contains("and.b32 %ur4, %ur4, 0xff;"));
        var uldcSubwordParamSection = new rtxdisam.SassSection();
        uldcSubwordParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ULDC.U16",
            Operands = { "UR4", "c[0x0][0x326]" },
        });
        string uldcSubwordParamPtx = rtxdisam.SassToPtx.Translate(
            uldcSubwordParamSection, "k",
            new[]
            {
                new rtxdisam.PtxParameter("p6", "u32", 0x324, 4),
                new rtxdisam.PtxParameter("p7", "u32", 0x326, 2),
            },
            "sm_86");
        Check("SASS to PTX ULDC.U16 exact sub-word parameter",
            uldcSubwordParamPtx.Contains("mov.u32 %ur4, %rpp1;") &&
            uldcSubwordParamPtx.Contains("and.b32 %ur4, %ur4, 0xffff;") &&
            !uldcSubwordParamPtx.Contains("shr.u32 %ur4, %rpp0, 16;"));
        var uldcSignedSection = new rtxdisam.SassSection();
        uldcSignedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ULDC.S8",
            Operands = { "UR4", "c[0x0][0x141]" },
        });
        string uldcSignedPtx = rtxdisam.SassToPtx.Translate(
            uldcSignedSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_86");
        Check("SASS to PTX ULDC.S8 sign-extends",
            uldcSignedPtx.Contains("shr.u32 %ur4, %ur4, 8;") &&
            uldcSignedPtx.Contains("shl.b32 %ur4, %ur4, 24;") &&
            uldcSignedPtx.Contains("shr.s32 %ur4, %ur4, 24;"));
        var conditionalExitSection = new rtxdisam.SassSection();
        conditionalExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
            Guard = "@P0",
            Operands = { "P1" },
        });
        string conditionalExitPtx = rtxdisam.SassToPtx.Translate(
            conditionalExitSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX guarded conditional EXIT",
            conditionalExitPtx.Contains("and.pred %pt0, %p0, %p1;") &&
            conditionalExitPtx.Contains("@%pt0 ret;"));
        var conditionalExitPlainSection = new rtxdisam.SassSection();
        conditionalExitPlainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
            Guard = "@P0",
            Operands = { "PT" },
        });
        string conditionalExitPlainPtx = rtxdisam.SassToPtx.Translate(
            conditionalExitPlainSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_80");
        Check("SASS to PTX guarded unconditional EXIT",
            conditionalExitPlainPtx.Contains("@%p0 ret;") &&
            !conditionalExitPlainPtx.Contains("and.pred"));
        var guardedIadd3Section = new rtxdisam.SassSection();
        guardedIadd3Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Guard = "@P0",
            Operands = { "R32", "P1", "R2", "R3", "RZ" },
        });
        guardedIadd3Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3.X",
            Guard = "@P0",
            Operands = { "R33", "R4", "R5", "RZ", "P1", "!PT" },
        });
        string guardedIadd3Ptx = rtxdisam.SassToPtx.Translate(
            guardedIadd3Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX guarded IADD3.X pair",
            guardedIadd3Ptx.Contains("add.u64") &&
            guardedIadd3Ptx.Contains("@%p0 mov.b64 {%r32, %r33}"));
        var lop3PredSection = new rtxdisam.SassSection();
        lop3PredSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP3.LUT",
            Operands = { "P0", "RZ", "R33", "0x7fffffff", "RZ", "0xc0", "!PT" },
        });
        string lop3PredPtx = rtxdisam.SassToPtx.Translate(
            lop3PredSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX LOP3 predicate destination",
            lop3PredPtx.Contains("setp.ne.u32 %p0"));
        var leaCarrySection = new rtxdisam.SassSection();
        leaCarrySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA",
            Operands = { "R19", "P1", "R29", "R10", "0x1" },
        });
        leaCarrySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3.X",
            Operands = { "R18", "RZ", "R9", "RZ", "P1", "!PT" },
        });
        string leaCarryPtx = rtxdisam.SassToPtx.Translate(
            leaCarrySection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX LEA/IADD3.X 64-bit pair",
            leaCarryPtx.Contains("add.u64") &&
            leaCarryPtx.Contains("mov.b64 {%r19, %r18}"));
        var fselInfSection = new rtxdisam.SassSection();
        fselInfSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FSEL",
            Operands = { "R33", "R30", "+INF", "P0" },
        });
        string fselInfPtx = rtxdisam.SassToPtx.Translate(
            fselInfSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX FSEL infinity",
            fselInfPtx.Contains("selp.f32") &&
            fselInfPtx.Contains("0x7f800000") &&
            fselInfPtx.Contains("mov.b32 %r33"));
        var guardedFselSection = new rtxdisam.SassSection();
        guardedFselSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FSEL",
            Guard = "@P0",
            Operands = { "R6", "RZ", "+INF", "!P2" },
        });
        string guardedFselPtx = rtxdisam.SassToPtx.Translate(
            guardedFselSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX guarded FSEL keeps destination",
            guardedFselPtx.Contains("@%p0 selp.f32") &&
            guardedFselPtx.Contains("@%p0 mov.b32 %r6"));
        var guardedMufuSection = new rtxdisam.SassSection();
        guardedMufuSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MUFU.EX2",
            Guard = "@P0",
            Operands = { "R0", "-|R3|" },
        });
        string guardedMufuPtx = rtxdisam.SassToPtx.Translate(
            guardedMufuSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX guarded MUFU keeps destination",
            guardedMufuPtx.Contains("@%p0 ex2.approx.f32") &&
            guardedMufuPtx.Contains("@%p0 mov.b32 %r0"));
        var f2iRoundSection = new rtxdisam.SassSection();
        f2iRoundSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.S32.F32",
            Operands = { "R1", "R2" },
        });
        f2iRoundSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.TRUNC.S32.F32",
            Operands = { "R3", "R2" },
        });
        f2iRoundSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.FLOOR.S32.F32",
            Operands = { "R5", "R2" },
        });
        string f2iRoundPtx = rtxdisam.SassToPtx.Translate(
            f2iRoundSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX F2I default rounds to nearest",
            f2iRoundPtx.Contains("cvt.rni.s32.f32 %r1") &&
            f2iRoundPtx.Contains("cvt.rzi.s32.f32 %r3") &&
            f2iRoundPtx.Contains("cvt.rmi.s32.f32 %r5"));
        var launchDimSection = new rtxdisam.SassSection();
        launchDimSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x0]" },
        });
        launchDimSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "c[0x0][0xc]" },
        });
        string launchDimPtx = rtxdisam.SassToPtx.Translate(
            launchDimSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX launch dimensions",
            launchDimPtx.Contains("%ntid.x") &&
            launchDimPtx.Contains("%nctaid.x"));
        var ushfHiSection = new rtxdisam.SassSection();
        ushfHiSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "USHF.R.U32.HI",
            Operands = { "UR4", "URZ", "0x1", "UR4" },
        });
        string ushfHiPtx = rtxdisam.SassToPtx.Translate(
            ushfHiSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX USHF funnel shift",
            ushfHiPtx.Contains("shf.r.wrap.b32 %ur4, %ur4, 0, 0x1;"));
        var voteuSection = new rtxdisam.SassSection();
        voteuSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VOTEU.ANY",
            Operands = { "UR4", "UPT", "PT" },
        });
        string voteuPtx = rtxdisam.SassToPtx.Translate(
            voteuSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX VOTEU",
            voteuPtx.Contains("vote.sync.any.pred") &&
            voteuPtx.Contains("selp.u32 %ur4"));
        var errbarSection = new rtxdisam.SassSection();
        errbarSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "ERRBAR" });
        string errbarPtx = rtxdisam.SassToPtx.Translate(
            errbarSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ERRBAR", errbarPtx.Contains("membar.cta"));
        var atomPredSection = new rtxdisam.SassSection();
        atomPredSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOMG.E.ADD.STRONG.GPU",
            Operands = { "PT", "R5", "[R4.64]", "R7" },
        });
        string atomPredPtx = rtxdisam.SassToPtx.Translate(
            atomPredSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ATOM with return predicate",
            atomPredPtx.Contains("atom.global.add.u32 %r5"));
        var sharedX4Section = new rtxdisam.SassSection();
        sharedX4Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[R3.X4+0x10]", "R16" },
        });
        string sharedX4Ptx = rtxdisam.SassToPtx.Translate(
            sharedX4Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX shared X4 address",
            sharedX4Ptx.Contains("st.shared.u32") &&
            sharedX4Ptx.Contains("shl.b32"));
        var bsyncCallSection = new rtxdisam.SassSection();
        bsyncCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Guard = "@P0",
            Operands = { "`(.L_bsync)`" },
        });
        bsyncCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_done)`" },
        });
        bsyncCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_bsync" });
        bsyncCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BSYNC",
            Operands = { "B0" },
        });
        bsyncCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_done" });
        bsyncCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string bsyncCallPtx = rtxdisam.SassToPtx.Translate(
            bsyncCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX guarded reconvergence call",
            Regex.IsMatch(bsyncCallPtx, @"@%p0 bra L_\d+;") &&
            !bsyncCallPtx.Contains("CALL"));
        var uaddWideSection = new rtxdisam.SassSection();
        uaddWideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UIADD3.64",
            Operands = { "UR4", "UP0", "UP1", "UR6", "UR8", "UR10" },
        });
        string uaddWidePtx = rtxdisam.SassToPtx.Translate(
            uaddWideSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX UIADD3.64",
            uaddWidePtx.Contains("add.u64") &&
            uaddWidePtx.Contains("mov.b64 {%ur4, %ur5}"));
        var imulWideSection = new rtxdisam.SassSection();
        imulWideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMUL.WIDE",
            Operands = { "R4", "P0", "R2", "R3" },
        });
        string imulWidePtx = rtxdisam.SassToPtx.Translate(
            imulWideSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IMUL.WIDE",
            imulWidePtx.Contains("mul.wide.s32 %rdt0, %r2, %r3;"));
        var imul32WideSection = new rtxdisam.SassSection();
        imul32WideSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMUL32I.WIDE",
            Operands = { "R4", "R2", "0x5" },
        });
        string imul32WidePtx = rtxdisam.SassToPtx.Translate(
            imul32WideSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IMUL32I.WIDE",
            imul32WidePtx.Contains("mul.wide.s32 %rdt0, %r2, 0x5;"));
        var iadd3PairSection = new rtxdisam.SassSection();
        iadd3PairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R12", "P0", "R8", "R10", "RZ" },
        });
        iadd3PairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.X",
            Operands = { "R13", "R9", "0x1", "R11", "P0" },
        });
        string iadd3PairPtx = rtxdisam.SassToPtx.Translate(
            iadd3PairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IADD3/IMAD.X 64-bit add",
            iadd3PairPtx.Contains("add.u64") &&
            iadd3PairPtx.Contains("mov.b64 {%r12, %r13}"));
        var iadd3XPairSection = new rtxdisam.SassSection();
        iadd3XPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R6", "P0", "R4", "R5", "RZ" },
        });
        iadd3XPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3.X",
            Operands = { "R7", "R4", "R5", "R6", "P0", "!PT" },
        });
        string iadd3XPairPtx = rtxdisam.SassToPtx.Translate(
            iadd3XPairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IADD3.X 64-bit pair",
            iadd3XPairPtx.Split("add.u64").Length >= 3);
        var iadd3GapSection = new rtxdisam.SassSection();
        iadd3GapSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R12", "P0", "R8", "R10", "RZ" },
        });
        iadd3GapSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "NOP" });
        iadd3GapSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.X",
            Operands = { "R13", "R9", "0x1", "R11", "P0" },
        });
        string iadd3GapPtx = rtxdisam.SassToPtx.Translate(
            iadd3GapSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IADD3 scheduled carry pair",
            iadd3GapPtx.Contains("add.u64"));
        // If the high half of a scheduled pair is still read before the .X
        // instruction, unpacking both halves at the low instruction would clobber
        // it.  The high half must be deferred to the .X node.
        var iadd3GapLiveSection = new rtxdisam.SassSection();
        iadd3GapLiveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R12", "P0", "R8", "R10", "RZ" },
        });
        iadd3GapLiveSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "NOP" });
        iadd3GapLiveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R14", "RZ", "R14", "R13", "RZ" },
        });
        iadd3GapLiveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.X",
            Operands = { "R13", "R9", "0x1", "R11", "P0" },
        });
        string iadd3GapLivePtx = rtxdisam.SassToPtx.Translate(
            iadd3GapLiveSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX defers a live scheduled carry pair",
            iadd3GapLivePtx.Contains("add.u64") &&
            !iadd3GapLivePtx.Contains("mov.b64 {%r12, %r13}") &&
            iadd3GapLivePtx.Contains("%r13}"));
        var isubLegacyPairSection = new rtxdisam.SassSection();
        isubLegacyPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R8.CC", "R2", "R4" },
        });
        isubLegacyPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB.X",
            Operands = { "R9", "R3", "R5" },
        });
        string isubLegacyPairPtx = rtxdisam.SassToPtx.Translate(
            isubLegacyPairSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_35");
        Check("SASS to PTX ISUB/ISUB.X 64-bit subtract",
            isubLegacyPairPtx.Contains("sub.u64") &&
            isubLegacyPairPtx.Contains("mov.b64 {%r8, %r9}"));
        var isubSm80PairSection = new rtxdisam.SassSection();
        isubSm80PairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R14", "P1", "R8", "-R10", "RZ" },
        });
        isubSm80PairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.X",
            Operands = { "R15", "R9", "0x1", "~R11", "P1" },
        });
        string isubSm80PairPtx = rtxdisam.SassToPtx.Translate(
            isubSm80PairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IADD3/IMAD.X 64-bit subtract",
            isubSm80PairPtx.Contains("sub.u64") &&
            isubSm80PairPtx.Contains("mov.b64 {%r14, %r15}"));
        var shiftLeftSection = new rtxdisam.SassSection();
        shiftLeftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.SHL.U32",
            Operands = { "R16", "R8", "0x8", "RZ" },
        });
        shiftLeftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L.U64.HI",
            Operands = { "R17", "R8", "0x3", "R9" },
        });
        string shiftLeftPtx = rtxdisam.SassToPtx.Translate(
            shiftLeftSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SHF.L.U64 pair",
            shiftLeftPtx.Contains("shl.b64") &&
            shiftLeftPtx.Contains("mov.b64 {%r16, %r17}"));
        var shiftRightSection = new rtxdisam.SassSection();
        shiftRightSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U64",
            Operands = { "R18", "R8", "0x5", "R9" },
        });
        shiftRightSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U32.HI",
            Operands = { "R19", "RZ", "0x5", "R9" },
        });
        string shiftRightPtx = rtxdisam.SassToPtx.Translate(
            shiftRightSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SHF.R.U64 pair",
            shiftRightPtx.Contains("shr.u64") &&
            shiftRightPtx.Contains("mov.b64 {%r18, %r19}"));
        var shiftRightGapSection = new rtxdisam.SassSection();
        shiftRightGapSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U64",
            Operands = { "R18", "R8", "0x5", "R9" },
        });
        shiftRightGapSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "NOP" });
        shiftRightGapSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U32.HI",
            Operands = { "R19", "RZ", "0x5", "R9" },
        });
        string shiftRightGapPtx = rtxdisam.SassToPtx.Translate(
            shiftRightGapSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX scheduled SHF.R.U64 pair",
            shiftRightGapPtx.Contains("shr.u64"));
        var shiftArithmeticSection = new rtxdisam.SassSection();
        shiftArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.S64",
            Operands = { "R20", "R8", "0x7", "R9" },
        });
        shiftArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.S32.HI",
            Operands = { "R21", "RZ", "0x7", "R9" },
        });
        string shiftArithmeticPtx = rtxdisam.SassToPtx.Translate(
            shiftArithmeticSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_80");
        Check("SASS to PTX SHF.R.S64 pair",
            shiftArithmeticPtx.Contains("shr.s64") &&
            shiftArithmeticPtx.Contains("mov.b64 {%r20, %r21}"));
        var dmulSection = new rtxdisam.SassSection();
        dmulSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DMUL",
            Operands = { "R4", "R6", "R8" },
        });
        string dmulPtx = rtxdisam.SassToPtx.Translate(
            dmulSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DMUL", dmulPtx.Contains("mul.f64"));
        var dfmaSection = new rtxdisam.SassSection();
        dfmaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DFMA",
            Operands = { "R4", "R6", "R8", "R10" },
        });
        string dfmaPtx = rtxdisam.SassToPtx.Translate(
            dfmaSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DFMA", dfmaPtx.Contains("fma.rn.f64"));
        var dfmaImmSection = new rtxdisam.SassSection();
        dfmaImmSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DFMA",
            Operands = { "R4", "R6", "2.2204460492503130808e-16", "R10" },
        });
        string dfmaImmPtx = rtxdisam.SassToPtx.Translate(
            dfmaImmSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DFMA double immediate",
            dfmaImmPtx.Contains("fma.rn.f64") &&
            dfmaImmPtx.Contains("0x3cb0000000000000"));
        var cs2rSection = new rtxdisam.SassSection();
        cs2rSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CS2R",
            Operands = { "R4", "SR_CLOCKLO" },
        });
        string cs2rPtx = rtxdisam.SassToPtx.Translate(
            cs2rSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX CS2R 64-bit clock",
            cs2rPtx.Contains("mov.u64") &&
            cs2rPtx.Contains("%clock64"));
        var cs2r32Section = new rtxdisam.SassSection();
        cs2r32Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CS2R.32",
            Operands = { "R6", "SR_CLOCKLO" },
        });
        string cs2r32Ptx = rtxdisam.SassToPtx.Translate(
            cs2r32Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX CS2R.32 clock",
            cs2r32Ptx.Contains("mov.u32 %r6, %clock;"));
        var dsetpSection = new rtxdisam.SassSection();
        dsetpSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DSETP.GT.AND",
            Operands = { "P0", "PT", "R4", "RZ", "PT" },
        });
        string dsetpPtx = rtxdisam.SassToPtx.Translate(
            dsetpSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DSETP f64 compare",
            dsetpPtx.Contains("mov.b64 %rdt0, {%r4, %r5};") &&
            dsetpPtx.Contains(
                "setp.gt.f64 %p0, %rdt0, 0d0000000000000000;"));
        var dsetpUnorderedSection = new rtxdisam.SassSection();
        dsetpUnorderedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DSETP.GTU.AND",
            Operands = { "P0", "PT", "R4", "R6", "PT" },
        });
        dsetpUnorderedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DSETP.GEU.AND",
            Operands = { "P1", "PT", "R8", "R10", "PT" },
        });
        string dsetpUnorderedPtx = rtxdisam.SassToPtx.Translate(
            dsetpUnorderedSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DSETP unordered compares",
            dsetpUnorderedPtx.Contains("setp.gtu.f64 %p0") &&
            dsetpUnorderedPtx.Contains("setp.geu.f64 %p1"));
        var dsetpMinSection = new rtxdisam.SassSection();
        dsetpMinSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DSETP.MIN.AND",
            Operands = { "P0", "P1", "R6", "R8", "PT" },
        });
        string dsetpMinPtx = rtxdisam.SassToPtx.Translate(
            dsetpMinSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DSETP.MIN",
            dsetpMinPtx.Contains("setp.lt.f64") &&
            dsetpMinPtx.Contains("setp.nan.f64") &&
            dsetpMinPtx.Contains("mov.pred %p1,"));
        var dsetpMaxSection = new rtxdisam.SassSection();
        dsetpMaxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DSETP.MAX.AND",
            Operands = { "P2", "P3", "R6", "R8", "PT" },
        });
        string dsetpMaxPtx = rtxdisam.SassToPtx.Translate(
            dsetpMaxSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DSETP.MAX",
            dsetpMaxPtx.Contains("setp.gt.f64") &&
            dsetpMaxPtx.Contains("setp.nan.f64") &&
            dsetpMaxPtx.Contains("mov.pred %p3,"));
        var multiParamSection = new rtxdisam.SassSection();
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "c[0x0][0x144]" },
        });
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R0", "R0", "c[0x0][0x148]" },
        });
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        multiParamSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string multiParamPtx = rtxdisam.SassToPtx.Translate(
            multiParamSection, "k",
            new[]
            {
                new rtxdisam.PtxParameter("p0", "u32", 0x148, 4),
                new rtxdisam.PtxParameter("p1", "u64", 0x140, 8),
            },
            "sm_50");
        Check("SASS to PTX multiple parameters",
            multiParamPtx.Contains("ld.param.u32 %rpp0, [p0];") &&
            multiParamPtx.Contains("ld.param.u64 %rdp1, [p1];") &&
            multiParamPtx.Contains("add.s32 %r0, %r0, %rpp0;"));
        // A by-value struct whose size is not a multiple of 8 (cudnn's
        // 260-byte GroupedDirectFpropParams) must still be exposed as a
        // parameter so later offsets stay aligned; dropping it made the
        // kernel read zeros.
        var wideStructSection = new rtxdisam.SassSection();
        wideStructSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x160]" },
        });
        wideStructSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "c[0x0][0x204]" },
        });
        wideStructSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R0", "R2", "R3" },
        });
        wideStructSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string wideStructPtx = rtxdisam.SassToPtx.Translate(
            wideStructSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64x33", 0x160, 260) },
            "sm_80");
        Check("SASS to PTX 260-byte by-value struct",
            wideStructPtx.Contains(".param .u64x33 p0") &&
            wideStructPtx.Contains("ld.param.u64 %rdp0, [p0];") &&
            wideStructPtx.Contains("[p0+164];"));
        // Magic division without the folded dividend add: ptxas emits
        // IMAD.HI.U32(n, magic, RZ) + SHF.R.U32.HI.  The lifter used to add the
        // dividend unconditionally, over-estimating every quotient.
        var umulhiDivSection = new rtxdisam.SassSection();
        umulhiDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R6", "0x5" },
        });
        umulhiDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.HI.U32",
            Operands = { "R2", "R6", "R13", "RZ" },
        });
        umulhiDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U32.HI",
            Operands = { "R4", "RZ", "0x1", "R2" },
        });
        umulhiDivSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string umulhiDivPtx = rtxdisam.SassToPtx.Translate(
            umulhiDivSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX umulhi division without add",
            umulhiDivPtx.Contains("mul.hi.u32 %r2, %r6, %r13;") &&
            !umulhiDivPtx.Contains("add.u32 %r2, %r2, %r6;"));
        // IMAD.HI.U32 reads a 64-bit addend pair {c, c+1}.  ATen's IntDivider
        // encodes the `+n` in the pair's high word (the low word is a fresh
        // zero register), so the lifter must add the high word, not the
        // register named by the SASS addend operand.
        var foldedAddDivSection = new rtxdisam.SassSection();
        foldedAddDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R7" },
        });
        foldedAddDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "RZ" },
        });
        foldedAddDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.HI.U32",
            Operands = { "R4", "R7", "R13", "R2" },
        });
        foldedAddDivSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.U32.HI",
            Operands = { "R5", "RZ", "0x5", "R4" },
        });
        foldedAddDivSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string foldedAddDivPtx = rtxdisam.SassToPtx.Translate(
            foldedAddDivSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX folded-add division",
            foldedAddDivPtx.Contains("mul.hi.u32") &&
            foldedAddDivPtx.Contains("add.cc.u32") &&
            foldedAddDivPtx.Contains("addc.u32") &&
            foldedAddDivPtx.Contains(", %r3;") &&
            !foldedAddDivPtx.Contains("add.u32 %r4, %r4, %r2;"));
        var unsupportedMadSection = new rtxdisam.SassSection();
        unsupportedMadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.MRG",
            Operands = { "R6", "R5", "R0.H1", "RZ" },
        });
        bool rejectedMad = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                unsupportedMadSection, "k",
                Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedMad = true;
        }
        Check("SASS to PTX rejects XMAD.MRG", rejectedMad);
        var xmadMultiplySection = new rtxdisam.SassSection();
        xmadMultiplySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R4", "R5", "R0", "RZ" },
        });
        xmadMultiplySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.MRG",
            Operands = { "R6", "R5", "R0.H1", "RZ" },
        });
        xmadMultiplySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.PSL.CBCC",
            Operands = { "R0", "R5.H1", "R6.H1", "R4" },
        });
        string xmadMultiplyPtx = rtxdisam.SassToPtx.Translate(
            xmadMultiplySection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX XMAD multiply sequence",
            xmadMultiplyPtx.Contains("mad.lo.s32 %r0, %r5, %r0, 0;") &&
            !xmadMultiplyPtx.Contains("XMAD.MRG"));
        var lop3LutSection = new rtxdisam.SassSection();
        lop3LutSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP3.LUT",
            Operands = { "R5", "R0", "c[0x0][0x148]", "R5", "0x1e" },
        });
        string lop3Ptx = rtxdisam.SassToPtx.Translate(
            lop3LutSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u32", 0x148, 4) },
            "sm_50");
        Check("SASS to PTX LOP3.LUT",
            lop3Ptx.Contains("not.b32") &&
            lop3Ptx.Contains("and.b32") &&
            lop3Ptx.Contains("or.b32") &&
            lop3Ptx.Contains("%rpp0"));
        var sharedSection = new rtxdisam.SassSection();
        sharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[RZ]", "R0" },
        });
        sharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS.U.32",
            Operands = { "R1", "[0x8]" },
        });
        string sharedPtx = rtxdisam.SassToPtx.Translate(
            sharedSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", null, 16);
        Check("SASS to PTX shared load/store",
            sharedPtx.Contains(".shared .align 16 .b8 __shared[16];") &&
            sharedPtx.Contains("st.shared.u32 [__shared], %r0;") &&
            sharedPtx.Contains("ld.shared.u32 %r1, [__shared+0x8];"));
        var vectorLoadSection = new rtxdisam.SassSection();
        vectorLoadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E.128",
            Operands = { "R4", "[R2]" },
        });
        vectorLoadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E.128",
            Operands = { "[R2]", "R4" },
        });
        string vectorPtx = rtxdisam.SassToPtx.Translate(
            vectorLoadSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX 128-bit vector load/store",
            vectorPtx.Contains(
                "ld.global.v4.u32 {%r4, %r5, %r6, %r7}, [%rdt0+0];") &&
            vectorPtx.Contains(
                "st.global.v4.u32 [%rdt1+0], {%r4, %r5, %r6, %r7};"));
        var vector64Section = new rtxdisam.SassSection();
        vector64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E.64",
            Operands = { "R4", "[R2]" },
        });
        string vector64Ptx = rtxdisam.SassToPtx.Translate(
            vector64Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX 64-bit vector load",
            vector64Ptx.Contains("ld.global.v2.u32 {%r4, %r5}, [%rdt0+0];"));
        var popcSection = new rtxdisam.SassSection();
        popcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "POPC",
            Operands = { "R0", "R1" },
        });
        string popcPtx = rtxdisam.SassToPtx.Translate(
            popcSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX POPC",
            popcPtx.Contains("popc.b32 %r0, %r1;"));
        var floU32Section = new rtxdisam.SassSection();
        floU32Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FLO.U32",
            Operands = { "R0", "R1" },
        });
        string floPtx = rtxdisam.SassToPtx.Translate(
            floU32Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FLO.U32",
            floPtx.Contains("clz.b32 %rt0, %r1;") &&
            floPtx.Contains("sub.u32 %r0, 31, %rt0;"));
        var shflSection = new rtxdisam.SassSection();
        shflSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHFL.UP",
            Operands = { "PT", "R0", "R1", "0x1", "0x1f" },
        });
        string shflPtx = rtxdisam.SassToPtx.Translate(
            shflSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SHFL.UP",
            shflPtx.Contains("activemask.b32 %rt0;") &&
            shflPtx.Contains(
                "shfl.sync.up.b32 %r0, %r1, 0x1, 0x1f, %rt0;"));
        var shfLeftSection = new rtxdisam.SassSection();
        shfLeftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.L.W",
            Operands = { "R6", "R0", "R7", "R6" },
        });
        string shfLeftPtx = rtxdisam.SassToPtx.Translate(
            shfLeftSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SHF.L",
            shfLeftPtx.Contains("shf.l.wrap.b32 %r6, %r6, %r0, %r7;"));
        var shfRightSection = new rtxdisam.SassSection();
        shfRightSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHF.R.W",
            Operands = { "R6", "R0", "R7", "R6" },
        });
        string shfRightPtx = rtxdisam.SassToPtx.Translate(
            shfRightSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SHF.R",
            shfRightPtx.Contains("shf.r.wrap.b32 %r6, %r0, %r6, %r7;"));
        var voteSection = new rtxdisam.SassSection();
        voteSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VOTE.ALL",
            Operands = { "P0", "P0" },
        });
        string votePtx = rtxdisam.SassToPtx.Translate(
            voteSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX VOTE.ALL",
            votePtx.Contains("activemask.b32 %rt0;") &&
            votePtx.Contains("vote.sync.all.pred %p0, %p0, %rt0;"));
        var ballotSection = new rtxdisam.SassSection();
        ballotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VOTE.ANY",
            Operands = { "R6", "PT", "P0" },
        });
        string ballotPtx = rtxdisam.SassToPtx.Translate(
            ballotSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX VOTE ballot",
            ballotPtx.Contains("vote.sync.ballot.b32 %r6, %p0, %rt0;"));
        var ballotNegSection = new rtxdisam.SassSection();
        ballotNegSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VOTE.ANY",
            Operands = { "R4", "PT", "!P2" },
        });
        string ballotNegPtx = rtxdisam.SassToPtx.Translate(
            ballotNegSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        // ptxas inverts the predicate for __ballot_sync's flag; dropping the
        // '!' turned topk's warp prefix scan upside down.
        Check("SASS to PTX VOTE negated ballot",
            ballotNegPtx.Contains("not.pred %pt0, %p2;") &&
            ballotNegPtx.Contains("vote.sync.ballot.b32 %r4, %pt0, %rt0;"));
        var atomAddSection = new rtxdisam.SassSection();
        atomAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOM.E.ADD",
            Operands = { "R6", "[R4]", "R2" },
        });
        string atomPtx = rtxdisam.SassToPtx.Translate(
            atomAddSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ATOM.ADD",
            atomPtx.Contains("atom.global.add.u32 %r6, [%rdt0+0], %r2;"));
        var sharedAtomSection = new rtxdisam.SassSection();
        sharedAtomSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOMS.ADD",
            Operands = { "R6", "[R7]", "R6" },
        });
        string sharedAtomPtx = rtxdisam.SassToPtx.Translate(
            sharedAtomSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", null, 16);
        Check("SASS to PTX ATOMS.ADD",
            sharedAtomPtx.Contains("mov.u64 %rdt1, __shared;") &&
            sharedAtomPtx.Contains("atom.shared.add.u32 %r6, [%rdt1], %r6;"));
        var atom64Section = new rtxdisam.SassSection();
        atom64Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOMG.E.ADD.64.STRONG.GPU",
            Operands = { "PT", "R4", "[R2.64]", "R4" },
        });
        string atom64Ptx = rtxdisam.SassToPtx.Translate(
            atom64Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ATOMG.ADD.64",
            atom64Ptx.Contains("atom.global.add.u64") &&
            atom64Ptx.Contains("mov.b64 {%r4, %r5}"));
        var atomsCastSection = new rtxdisam.SassSection();
        atomsCastSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOMS.CAST.SPIN.64",
            Operands = { "R6", "[RZ]", "R4", "R6" },
        });
        string atomsCastPtx = rtxdisam.SassToPtx.Translate(
            atomsCastSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX ATOMS.CAST.SPIN.64",
            atomsCastPtx.Contains("atom.shared.cas.b64") &&
            atomsCastPtx.Contains("setp.ne.u64") &&
            atomsCastPtx.Contains("bra L_cas_spin_"));
        var ltmaskSection = new rtxdisam.SassSection();
        ltmaskSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R3", "SR_LTMASK" },
        });
        string ltmaskPtx = rtxdisam.SassToPtx.Translate(
            ltmaskSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SR_LTMASK",
            ltmaskPtx.Contains("mov.u32 %r3, %lanemask_lt;"));
        var lemaskSection = new rtxdisam.SassSection();
        lemaskSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R3", "SR_LEMASK" },
        });
        string lemaskPtx = rtxdisam.SassToPtx.Translate(
            lemaskSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        // SR_LEMASK is %lanemask_le (the lane mask ptxas uses in warp
        // prefix scans), not the active-lane mask.
        Check("SASS to PTX SR_LEMASK",
            lemaskPtx.Contains("mov.u32 %r3, %lanemask_le;"));
        var addressPairSection = new rtxdisam.SassSection();
        addressPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4.CC", "R6", "c[0x0][0x140]" },
        });
        addressPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R5", "R0", "c[0x0][0x144]" },
        });
        addressPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R4", "[R4]" },
        });
        string addressPairPtx = rtxdisam.SassToPtx.Translate(
            addressPairSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX 64-bit address pair",
            addressPairPtx.Contains("mov.b64 %rdt0, {%r6, %r0};") &&
            addressPairPtx.Contains("mov.b64 %rdt1, %rpptr0;") &&
            addressPairPtx.Contains("add.u64 %rdt0, %rdt0, %rdt1;") &&
            addressPairPtx.Contains("ld.global.u32 %r4, [%rdt0+0];"));
        var genericAddPairSection = new rtxdisam.SassSection();
        genericAddPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R2.CC", "R5", "0x4" },
        });
        genericAddPairSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R3", "R6", "RZ" },
        });
        genericAddPairSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string genericAddPairPtx = rtxdisam.SassToPtx.Translate(
            genericAddPairSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX generic add pair",
            genericAddPairPtx.Contains("mov.b64 %rdt0, {%r5, %r6};") &&
            genericAddPairPtx.Contains("add.u64 %rdt1, %rdt0, 0x4;") &&
            genericAddPairPtx.Contains("cvta.to.global.u64 %rdt"));
        var pointerAddSection = new rtxdisam.SassSection();
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "c[0x0][0x144]" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "R2" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "R3" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R6.CC", "R4", "0x4" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R7", "R5", "RZ" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R6]" },
        });
        pointerAddSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string pointerAddPtx = rtxdisam.SassToPtx.Translate(
            pointerAddSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX pointer add pair",
            pointerAddPtx.Contains("add.u64 %rdt") &&
            pointerAddPtx.Contains(", %rpptr0, 0x4;") &&
            pointerAddPtx.Contains("cvta.to.global.u64 %rdt") &&
            pointerAddPtx.Contains("ld.global.u32 %r0, [%rdt"));
        var leaPointerSection = new rtxdisam.SassSection();
        leaPointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "c[0x0][0x140]" },
        });
        leaPointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA",
            Operands = { "R2.CC", "R0", "RZ" },
        });
        leaPointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA.HI.X",
            Operands = { "P0", "R3", "R0", "RZ", "R3" },
        });
        leaPointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        string leaPointerPtx = rtxdisam.SassToPtx.Translate(
            leaPointerSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX LEA pointer elision",
            leaPointerPtx.Contains("ld.global.u32 %r0, [%rpptr0+0];") &&
            !leaPointerPtx.Contains("LEA"));
        var pointerImmediateSection = new rtxdisam.SassSection();
        pointerImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x140]" },
        });
        pointerImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R4.CC", "R4", "0x4" },
        });
        pointerImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R5", "RZ", "c[0x0][0x144]" },
        });
        pointerImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R4", "[R4]" },
        });
        string pointerImmediatePtx = rtxdisam.SassToPtx.Translate(
            pointerImmediateSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX pointer plus immediate",
            pointerImmediatePtx.Contains("mov.b64 %rdt") &&
            pointerImmediatePtx.Contains(", %rpptr0;") &&
            pointerImmediatePtx.Contains("add.u64 %rdt") &&
            pointerImmediatePtx.Contains(", 0x4;") &&
            pointerImmediatePtx.Contains("ld.global.u32 %r4, [%rdt"));
        var paramValueSection = new rtxdisam.SassSection();
        paramValueSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "c[0x0][0x140]" },
        });
        paramValueSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP32I.AND",
            Operands = { "R0", "R0", "0xffffff" },
        });
        paramValueSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string paramValuePtx = rtxdisam.SassToPtx.Translate(
            paramValueSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX parameter low word value",
            paramValuePtx.Contains("mov.b64 {%ru0, %ru1}, %rdp0;") &&
            paramValuePtx.Contains("mov.b32 %r0, %ru0;") &&
            paramValuePtx.Contains("and.b32 %r0, %r0, 0xffffff;"));
        var paramHighValueSection = new rtxdisam.SassSection();
        paramHighValueSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "c[0x0][0x144]" },
        });
        paramHighValueSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LOP32I.AND",
            Operands = { "R3", "R3", "0x1" },
        });
        paramHighValueSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string paramHighValuePtx = rtxdisam.SassToPtx.Translate(
            paramHighValueSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX parameter high word value",
            paramHighValuePtx.Contains("mov.b64 {%ru0, %ru1}, %rdp0;") &&
            paramHighValuePtx.Contains("mov.b32 %r3, %ru1;") &&
            paramHighValuePtx.Contains("and.b32 %r3, %r3, 0x1;"));
        var imadAddressSection = new rtxdisam.SassSection();
        imadAddressSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISCADD",
            Operands = { "R4.CC", "R0", "c[0x0][0x140]", "0x4" },
        });
        imadAddressSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD.U32.U32.HI.X",
            Operands = { "R5", "R0", "R3", "c[0x0][0x144]" },
        });
        imadAddressSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E.128",
            Operands = { "R4", "[R4]" },
        });
        string imadAddressPtx = rtxdisam.SassToPtx.Translate(
            imadAddressSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX ISCADD/IMAD address pair",
            imadAddressPtx.Contains("mul.wide.u32 %rdt0, %r0, %r3;") &&
            imadAddressPtx.Contains("mov.b64 %rdt1, %rpptr0;") &&
            imadAddressPtx.Contains("add.u64 %rdt0, %rdt0, %rdt1;") &&
            imadAddressPtx.Contains(
                "ld.global.v4.u32 {%r4, %r5, %r6, %r7}, [%rdt0+0];"));
        var xmadInterveningSection = new rtxdisam.SassSection();
        xmadInterveningSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R8", "R2", "R4", "RZ" },
        });
        xmadInterveningSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.MRG",
            Operands = { "R9", "R2", "R4.H1", "RZ" },
        });
        xmadInterveningSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x148]" },
        });
        xmadInterveningSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.PSL.CBCC",
            Operands = { "R6", "R2.H1", "R9.H1", "R8" },
        });
        string xmadInterveningPtx = rtxdisam.SassToPtx.Translate(
            xmadInterveningSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x148, 8) },
            "sm_50");
        Check("SASS to PTX XMAD with intervening MOV",
            xmadInterveningPtx.Contains("mad.lo.s32 %r6, %r2, %r4, 0;") &&
            !xmadInterveningPtx.Contains("XMAD.MRG"));
        var xmadShflSection = new rtxdisam.SassSection();
        xmadShflSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.MRG",
            Operands = { "R9", "R2", "R8.H1", "RZ" },
        });
        xmadShflSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHFL.IDX",
            Operands = { "PT", "R7", "R6", "R7", "0x1f" },
        });
        xmadShflSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD",
            Operands = { "R8", "R2", "R8", "R7" },
        });
        xmadShflSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XMAD.PSL.CBCC",
            Operands = { "R2", "R2.H1", "R9.H1", "R8" },
        });
        string xmadShflPtx = rtxdisam.SassToPtx.Translate(
            xmadShflSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX interleaved XMAD multiply",
            xmadShflPtx.Contains(
                "shfl.sync.idx.b32 %r7, %r6, %r7, 0x1f, %rt0;") &&
            xmadShflPtx.Contains("mad.lo.s32 %r2, %r2, %r8, %r7;"));
        var dynamicSharedSection = new rtxdisam.SassSection();
        dynamicSharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[R6]", "R0" },
        });
        string dynamicSharedPtx = rtxdisam.SassToPtx.Translate(
            dynamicSharedSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", null, 16);
        Check("SASS to PTX dynamic shared address",
            dynamicSharedPtx.Contains("cvt.u64.u32 %rdt0, %r6;") &&
            dynamicSharedPtx.Contains("mov.u64 %rdt1, __shared;") &&
            dynamicSharedPtx.Contains("add.u64 %rdt1, %rdt1, %rdt0;") &&
            dynamicSharedPtx.Contains("st.shared.u32 [%rdt1], %r0;"));
        var externSharedSection = new rtxdisam.SassSection();
        externSharedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS.128",
            Operands = { "[R2.X16+0x10]", "R4" },
        });
        string externSharedPtx = rtxdisam.SassToPtx.Translate(
            externSharedSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", null, 16);
        Check("SASS to PTX extern shared displacement",
            externSharedPtx.Contains(".shared .align 16 .b8 __shared[16];") &&
            externSharedPtx.Contains(
                ".extern .shared .align 16 .b8 __shared_dynamic[];") &&
            externSharedPtx.Contains(", __shared_dynamic;") &&
            externSharedPtx.Contains("st.shared.v4.u32 [%rdt"));
        var foldedDynamicSection = new rtxdisam.SassSection();
        foldedDynamicSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LEA",
            Operands = { "R3", "R2", "0x10", "0x4" },
        });
        foldedDynamicSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[R3]", "R0" },
        });
        string foldedDynamicPtx = rtxdisam.SassToPtx.Translate(
            foldedDynamicSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", null, 16);
        Check("SASS to PTX extern shared folded base",
            foldedDynamicPtx.Contains(", __shared_dynamic;") &&
            foldedDynamicPtx.Contains("-0x10], %r0;"));
        var selNegatedSection = new rtxdisam.SassSection();
        selNegatedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SEL",
            Operands = { "R6", "RZ", "0x1", "!P0" },
        });
        string selNegatedPtx = rtxdisam.SassToPtx.Translate(
            selNegatedSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SEL negated predicate",
            selNegatedPtx.Contains("selp.b32 %r6, 0x1, 0, %p0;"));
        var isetpUnsignedSection = new rtxdisam.SassSection();
        isetpUnsignedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.NE.U32.AND",
            Operands = { "P0", "PT", "R2", "RZ", "PT" },
        });
        string isetpUnsignedPtx = rtxdisam.SassToPtx.Translate(
            isetpUnsignedSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISETP unsigned",
            isetpUnsignedPtx.Contains("setp.ne.u32 %p0, %r2, 0;"));
        var isetpOrChainSection = new rtxdisam.SassSection();
        isetpOrChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.EQ.OR",
            Operands = { "P0", "PT", "R16", "RZ", "P0" },
        });
        string isetpOrChainPtx = rtxdisam.SassToPtx.Translate(
            isetpOrChainSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISETP OR combine",
            isetpOrChainPtx.Contains("setp.eq.s32 %pt0, %r16, 0;") &&
            isetpOrChainPtx.Contains("or.pred %p0, %pt0, %p0;"));
        var isetpAndChainSection = new rtxdisam.SassSection();
        isetpAndChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.NE.AND",
            Operands = { "P1", "PT", "R6", "RZ", "!P5" },
        });
        string isetpAndChainPtx = rtxdisam.SassToPtx.Translate(
            isetpAndChainSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISETP AND combine",
            isetpAndChainPtx.Contains("setp.ne.s32 %pt0, %r6, 0;") &&
            isetpAndChainPtx.Contains("not.pred %pt1, %p5;") &&
            isetpAndChainPtx.Contains("and.pred %p1, %pt0, %pt1;"));
        var i2fRoundSection = new rtxdisam.SassSection();
        i2fRoundSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.U32.RP",
            Operands = { "R16", "R7" },
        });
        i2fRoundSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F32.RM",
            Operands = { "R17", "R8" },
        });
        string i2fRoundPtx = rtxdisam.SassToPtx.Translate(
            i2fRoundSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX I2F rounding modes",
            i2fRoundPtx.Contains("cvt.rp.f32.u32 %r16, %r7;") &&
            i2fRoundPtx.Contains("cvt.rm.f32.s32 %r17, %r8;"));

        var isetpExChainSection = new rtxdisam.SassSection();
        isetpExChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.OR.EX",
            Operands = { "P0", "PT", "R17", "R19", "P0", "P1" },
        });
        string isetpExChainPtx = rtxdisam.SassToPtx.Translate(
            isetpExChainSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISETP.EX chains source predicate",
            isetpExChainPtx.Contains("setp.lt.s32 %pt1, %r17, %r19;") &&
            isetpExChainPtx.Contains("or.pred %p0, %pt0, %p0;"));
        var isetpExPlainSection = new rtxdisam.SassSection();
        isetpExPlainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GE.AND.EX",
            Operands = { "P0", "PT", "R11", "R9", "PT", "P0" },
        });
        string isetpExPlainPtx = rtxdisam.SassToPtx.Translate(
            isetpExPlainSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISETP.EX plain chain",
            isetpExPlainPtx.Contains("mov.pred %p0, %pt0;"));
        var uisetpChainSection = new rtxdisam.SassSection();
        uisetpChainSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UISETP.NE.U32.AND",
            Operands = { "UP0", "UPT", "UR6", "URZ", "UP0" },
        });
        string uisetpChainPtx = rtxdisam.SassToPtx.Translate(
            uisetpChainSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX UISETP chains source predicate",
            uisetpChainPtx.Contains("setp.ne.u32 %pt0, %ur6, 0;") &&
            uisetpChainPtx.Contains("and.pred %up0, %pt0, %up0;"));
        var uisetpExSection = new rtxdisam.SassSection();
        uisetpExSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "UISETP.GE.AND.EX",
            Operands = { "UP0", "UPT", "UR4", "UR15", "UPT", "UP0" },
        });
        string uisetpExPtx = rtxdisam.SassToPtx.Translate(
            uisetpExSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX UISETP.EX 64-bit compare",
            uisetpExPtx.Contains("setp.gt.s32 %pt1, %ur4, %ur15;") &&
            uisetpExPtx.Contains("mov.pred %up0, %pt0;"));
        var icmpSection = new rtxdisam.SassSection();
        icmpSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ICMP.LE",
            Operands = { "R6", "R0", "0x2", "R2" },
        });
        string icmpPtx = rtxdisam.SassToPtx.Translate(
            icmpSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ICMP select",
            icmpPtx.Contains("setp.le.s32 %pt0, %r2, 0;") &&
            icmpPtx.Contains("selp.b32 %r6, %r0, 0x2, %pt0;"));
        var fmnmxMinSection = new rtxdisam.SassSection();
        fmnmxMinSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FMNMX",
            Operands = { "R7", "R0", "R6", "PT" },
        });
        string fmnmxMinPtx = rtxdisam.SassToPtx.Translate(
            fmnmxMinSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FMNMX min",
            fmnmxMinPtx.Contains("mov.b32 %ft0, %r0;") &&
            fmnmxMinPtx.Contains("mov.b32 %ft1, %r6;") &&
            fmnmxMinPtx.Contains("min.f32 %ft2, %ft0, %ft1;") &&
            fmnmxMinPtx.Contains("mov.b32 %r7, %ft2;"));
        var fmnmxMaxSection = new rtxdisam.SassSection();
        fmnmxMaxSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FMNMX",
            Operands = { "R7", "R0", "R6", "!PT" },
        });
        string fmnmxMaxPtx = rtxdisam.SassToPtx.Translate(
            fmnmxMaxSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FMNMX max",
            fmnmxMaxPtx.Contains("max.f32 %ft2, %ft0, %ft1;") &&
            fmnmxMaxPtx.Contains("mov.b32 %r7, %ft2;"));
        var daddSection = new rtxdisam.SassSection();
        daddSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DADD",
            Operands = { "R4", "R2", "3.25" },
        });
        string daddPtx = rtxdisam.SassToPtx.Translate(
            daddSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX DADD",
            daddPtx.Contains("mov.b64 %rdt0, {%r2, %r3};") &&
            daddPtx.Contains("mov.b64 %rdt1, 0x400a000000000000;") &&
            daddPtx.Contains("add.f64 %dt0, %rdt0, %rdt1;") &&
            daddPtx.Contains("mov.b64 {%r4, %r5}, %dt0;"));
        var daddNegSection = new rtxdisam.SassSection();
        daddNegSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DADD",
            Operands = { "R4", "R2", "-R6" },
        });
        string daddNegPtx = rtxdisam.SassToPtx.Translate(
            daddNegSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DADD negated register",
            daddNegPtx.Contains("neg.f64") &&
            daddNegPtx.Contains("add.f64"));
        var mufuSinSection = new rtxdisam.SassSection();
        mufuSinSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MUFU.SIN",
            Operands = { "R6", "R6" },
        });
        string mufuPtx = rtxdisam.SassToPtx.Translate(
            mufuSinSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX MUFU.SIN",
            mufuPtx.Contains("mul.f32") &&
            mufuPtx.Contains("0f40c90fdb") &&
            mufuPtx.Contains("sin.approx.f32"));
        var mufuTanhSection = new rtxdisam.SassSection();
        mufuTanhSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MUFU.TANH",
            Operands = { "R6", "R7" },
        });
        string mufuTanhPtx = rtxdisam.SassToPtx.Translate(
            mufuTanhSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX MUFU.TANH",
            mufuTanhPtx.Contains("tanh.approx.f32"));
        var hmmaSection = new rtxdisam.SassSection();
        hmmaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "HMMA.16816.F32",
            Operands = { "R4", "R4", "R2", "RZ" },
        });
        string hmmaPtx = rtxdisam.SassToPtx.Translate(
            hmmaSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX HMMA.16816",
            hmmaPtx.Contains(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        var hmmaTf32Section = new rtxdisam.SassSection();
        hmmaTf32Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "HMMA.1684.F32.TF32",
            Operands = { "R12", "R12", "RZ", "RZ" },
        });
        string hmmaTf32Ptx = rtxdisam.SassToPtx.Translate(
            hmmaTf32Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX HMMA.1684.TF32",
            hmmaTf32Ptx.Contains(
                "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"));
        var immaSection = new rtxdisam.SassSection();
        immaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMMA.8816.S8.S8",
            Operands = { "R8", "RZ.ROW", "RZ.COL", "RZ" },
        });
        string immaPtx = rtxdisam.SassToPtx.Translate(
            immaSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IMMA.8816",
            immaPtx.Contains(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"));
        var dmmaSection = new rtxdisam.SassSection();
        dmmaSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DMMA.884",
            Operands = { "R4", "RZ", "RZ", "RZ" },
        });
        string dmmaPtx = rtxdisam.SassToPtx.Translate(
            dmmaSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DMMA.884",
            dmmaPtx.Contains(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"));
        var idp4Section = new rtxdisam.SassSection();
        idp4Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IDP.4A.S8.U8",
            Operands = { "R9", "R0", "R5", "R8" },
        });
        string idp4Ptx = rtxdisam.SassToPtx.Translate(
            idp4Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IDP.4A",
            idp4Ptx.Contains("dp4a.s32.u32 %r9, %r0, %r5, %r8;"));
        var idp2Section = new rtxdisam.SassSection();
        idp2Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IDP.2A.HI.S16.S8",
            Operands = { "R19", "R0", "R5", "R8" },
        });
        string idp2Ptx = rtxdisam.SassToPtx.Translate(
            idp2Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IDP.2A",
            idp2Ptx.Contains("dp2a.hi.s32.s32 %r19, %r0, %r5, %r8;"));
        var rroSection = new rtxdisam.SassSection();
        rroSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "RRO.SINCOS",
            Operands = { "R6", "R2" },
        });
        string rroPtx = rtxdisam.SassToPtx.Translate(
            rroSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX RRO.SINCOS",
            rroPtx.Contains("mul.f32 %r6, %r2, 0.15915493667125701904;"));
        var imul32iSection = new rtxdisam.SassSection();
        imul32iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMUL32I",
            Operands = { "R2", "R0", "0x3" },
        });
        imul32iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMUL32I.HI",
            Operands = { "R3", "R0", "0x3" },
        });
        string imul32iPtx = rtxdisam.SassToPtx.Translate(
            imul32iSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX IMUL32I",
            imul32iPtx.Contains("mul.lo.s32 %r2, %r0, 0x3;") &&
            imul32iPtx.Contains("mul.hi.s32 %r3, %r0, 0x3;"));
        var iadd3Section = new rtxdisam.SassSection();
        iadd3Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD3",
            Operands = { "R2", "R0", "R1", "R3" },
        });
        string iadd3Ptx = rtxdisam.SassToPtx.Translate(
            iadd3Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX IADD3",
            iadd3Ptx.Contains("add.s32 %rt0, %r0, %r1;") &&
            iadd3Ptx.Contains("add.s32 %r2, %rt0, %r3;"));
        var bfeSection = new rtxdisam.SassSection();
        bfeSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BFE.U32",
            Operands = { "R7", "R0", "0x305" },
        });
        string bfePtx = rtxdisam.SassToPtx.Translate(
            bfeSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX BFE",
            bfePtx.Contains("bfe.u32 %r7, %r0, 5, 3;"));
        var bfiSection = new rtxdisam.SassSection();
        bfiSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BFI",
            Operands = { "R9", "R0", "0x305", "R6" },
        });
        string bfiPtx = rtxdisam.SassToPtx.Translate(
            bfiSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX BFI",
            bfiPtx.Contains("bfi.b32 %r9, %r0, %r6, 5, 3;"));
        var bmskSection = new rtxdisam.SassSection();
        bmskSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BMSK",
            Operands = { "R4", "R2", "R3" },
        });
        bmskSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BMSK.C",
            Operands = { "R5", "R2", "R3" },
        });
        string bmskPtx = rtxdisam.SassToPtx.Translate(
            bmskSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX BMSK",
            bmskPtx.Contains("bmsk.wrap.b32 %r4, %r2, %r3;") &&
            bmskPtx.Contains("bmsk.clamp.b32 %r5, %r2, %r3;"));
        var sgxtSection = new rtxdisam.SassSection();
        sgxtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SGXT",
            Operands = { "R7", "R7", "0x8" },
        });
        sgxtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SGXT.U32",
            Operands = { "R8", "R6", "R7" },
        });
        string sgxtPtx = rtxdisam.SassToPtx.Translate(
            sgxtSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SGXT",
            sgxtPtx.Contains("bfe.s32 %r7, %r7, 0, 0x8;") &&
            sgxtPtx.Contains("bfe.u32 %r8, %r6, 0, %r7;"));
        var iabsSection = new rtxdisam.SassSection();
        iabsSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IABS",
            Operands = { "R9", "R0" },
        });
        string iabsPtx = rtxdisam.SassToPtx.Translate(
            iabsSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX IABS", iabsPtx.Contains("abs.s32 %r9, %r0;"));
        var negSection = new rtxdisam.SassSection();
        negSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NEG",
            Operands = { "R9", "R0" },
        });
        string negPtx = rtxdisam.SassToPtx.Translate(
            negSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX NEG", negPtx.Contains("neg.s32 %r9, %r0;"));
        var sadSection = new rtxdisam.SassSection();
        sadSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SAD",
            Operands = { "R5", "R0", "R5", "R4" },
        });
        string sadPtx = rtxdisam.SassToPtx.Translate(
            sadSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SAD",
            sadPtx.Contains("abs.s32") &&
            sadPtx.Contains("add.s32 %r5"));
        var divSection = new rtxdisam.SassSection();
        divSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DIV.S32",
            Operands = { "R7", "R0", "R5" },
        });
        divSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "REM.U32",
            Operands = { "R8", "R0", "R5" },
        });
        divSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "DIV.S64",
            Operands = { "R10", "R0", "R5" },
        });
        string divPtx = rtxdisam.SassToPtx.Translate(
            divSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX DIV/REM",
            divPtx.Contains("div.s32 %r7, %r0, %r5;") &&
            divPtx.Contains("rem.u32 %r8, %r0, %r5;") &&
            divPtx.Contains("div.s64"));
        var bitArithmeticSection = new rtxdisam.SassSection();
        bitArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CLZ",
            Operands = { "R9", "R0" },
        });
        bitArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BFIND.U32",
            Operands = { "R10", "R0" },
        });
        bitArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BREV",
            Operands = { "R11", "R0" },
        });
        bitArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FNS",
            Operands = { "R12", "R0", "R5", "R6" },
        });
        bitArithmeticSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CLMAD.LO",
            Operands = { "R14", "R0", "R6", "R8" },
        });
        string bitArithmeticPtx = rtxdisam.SassToPtx.Translate(
            bitArithmeticSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX bit arithmetic",
            bitArithmeticPtx.Contains("clz.b32 %r9, %r0;") &&
            bitArithmeticPtx.Contains("bfind.u32 %r10, %r0;") &&
            bitArithmeticPtx.Contains("brev.b32 %r11, %r0;") &&
            bitArithmeticPtx.Contains("fns.b32 %r12, %r0, %r5, %r6;") &&
            bitArithmeticPtx.Contains("clmad.lo.u64"));
        var floatAliasSection = new rtxdisam.SassSection();
        void AddFloatAlias(string mnemonic, params string[] operands)
        {
            var alias = new rtxdisam.SassInstruction { Mnemonic = mnemonic };
            alias.Operands.AddRange(operands);
            floatAliasSection.Nodes.Add(alias);
        }
        AddFloatAlias("ADD.F32", "R1", "R2", "R3");
        AddFloatAlias("SUB.F32", "R4", "R2", "R3");
        AddFloatAlias("MUL.F32", "R5", "R2", "R3");
        AddFloatAlias("FMA.F32", "R6", "R2", "R3", "R4");
        AddFloatAlias("DIV.F32", "R7", "R2", "R3");
        AddFloatAlias("ABS.F32", "R8", "R2");
        AddFloatAlias("NEG.F32", "R9", "R2");
        AddFloatAlias("MIN.F32", "R10", "R2", "R3");
        AddFloatAlias("MAX.F32", "R11", "R2", "R3");
        AddFloatAlias("RCP.F32", "R12", "R2");
        AddFloatAlias("SQRT.F32", "R13", "R2");
        AddFloatAlias("RSQRT.F32", "R14", "R2");
        AddFloatAlias("SIN.F32", "R15", "R2");
        AddFloatAlias("COS.F32", "R16", "R2");
        AddFloatAlias("LG2.F32", "R17", "R2");
        AddFloatAlias("EX2.F32", "R19", "R2");
        AddFloatAlias("TANH.F32", "R20", "R2");
        AddFloatAlias("COPYSIGN.F32", "R18", "R2", "R3");
        AddFloatAlias("TESTP.FINITE.F32", "P0", "R2");
        string floatAliasPtx = rtxdisam.SassToPtx.Translate(
            floatAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX float aliases",
            floatAliasPtx.Contains("add.f32") &&
            floatAliasPtx.Contains("sub.f32") &&
            floatAliasPtx.Contains("mul.f32") &&
            floatAliasPtx.Contains("fma.rn.f32") &&
            floatAliasPtx.Contains("div.rn.f32") &&
            floatAliasPtx.Contains("abs.f32") &&
            floatAliasPtx.Contains("neg.f32") &&
            floatAliasPtx.Contains("min.f32") &&
            floatAliasPtx.Contains("max.f32") &&
            floatAliasPtx.Contains("rcp.approx.f32") &&
            floatAliasPtx.Contains("sqrt.rn.f32") &&
            floatAliasPtx.Contains("rsqrt.approx.f32") &&
            floatAliasPtx.Contains("sin.approx.f32") &&
            floatAliasPtx.Contains("cos.approx.f32") &&
            floatAliasPtx.Contains("lg2.approx.f32") &&
            floatAliasPtx.Contains("ex2.approx.f32") &&
            floatAliasPtx.Contains("tanh.approx.f32") &&
            floatAliasPtx.Contains("copysign.f32") &&
            floatAliasPtx.Contains("testp.finite.f32"));
        var ffmaReuseNegateSection = new rtxdisam.SassSection();
        ffmaReuseNegateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FFMA",
            Operands = { "R8", "R17", "-R0.reuse", "0.14084610342979431152" },
        });
        string ffmaReuseNegatePtx = rtxdisam.SassToPtx.Translate(
            ffmaReuseNegateSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX FFMA negated reuse source",
            ffmaReuseNegatePtx.Contains("neg.f32 %ft0, %r0;") &&
            !ffmaReuseNegatePtx.Contains("neg.s32"));
        var float64AliasSection = new rtxdisam.SassSection();
        void AddFloat64Alias(string mnemonic, params string[] operands)
        {
            var alias = new rtxdisam.SassInstruction { Mnemonic = mnemonic };
            alias.Operands.AddRange(operands);
            float64AliasSection.Nodes.Add(alias);
        }
        AddFloat64Alias("ADD.F64", "R1", "R2", "R4");
        AddFloat64Alias("SUB.F64", "R6", "R2", "R4");
        AddFloat64Alias("MUL.F64", "R8", "R2", "R4");
        AddFloat64Alias("FMA.F64", "R10", "R2", "R4", "R6");
        AddFloat64Alias("MIN.F64", "R12", "R2", "R4");
        AddFloat64Alias("MAX.F64", "R14", "R2", "R4");
        string float64AliasPtx = rtxdisam.SassToPtx.Translate(
            float64AliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX float64 aliases",
            float64AliasPtx.Contains("add.f64") &&
            float64AliasPtx.Contains("sub.f64") &&
            float64AliasPtx.Contains("mul.f64") &&
            float64AliasPtx.Contains("fma.rn.f64") &&
            float64AliasPtx.Contains("min.f64") &&
            float64AliasPtx.Contains("max.f64"));
        var ffmaModifierSection = new rtxdisam.SassSection();
        ffmaModifierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FFMA.FTZ.SAT",
            Operands = { "R1", "R2", "R3", "R4" },
        });
        string ffmaModifierPtx = rtxdisam.SassToPtx.Translate(
            ffmaModifierSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX FFMA FTZ/SAT",
            ffmaModifierPtx.Contains("fma.rn.ftz.sat.f32"));
        var cvtAliasSection = new rtxdisam.SassSection();
        cvtAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CVT.RN.F32.S32",
            Operands = { "R1", "R2" },
        });
        cvtAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CVT.RZI.S64.F64",
            Operands = { "R4", "R6" },
        });
        cvtAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2F.F64.S32",
            Operands = { "R8", "R2" },
        });
        cvtAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2FP.F32.U32",
            Operands = { "R9", "R2" },
        });
        cvtAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "F2I.U64.F64",
            Operands = { "R10", "R6" },
        });
        string cvtAliasPtx = rtxdisam.SassToPtx.Translate(
            cvtAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX CVT aliases",
            cvtAliasPtx.Contains("cvt.rn.f32.s32") &&
            cvtAliasPtx.Contains("cvt.rzi.s64.f64") &&
            cvtAliasPtx.Contains("cvt.rn.f64.s32") &&
            cvtAliasPtx.Contains("cvt.rn.f32.u32 %r9, %r2;") &&
            // F2I's implied Round3 default is round-to-nearest, not truncation.
            cvtAliasPtx.Contains("cvt.rni.u64.f64"));
        var vabsdiffSection = new rtxdisam.SassSection();
        vabsdiffSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VABSDIFF",
            Operands = { "R5", "R0", "R5", "R4" },
        });
        string vabsdiffPtx = rtxdisam.SassToPtx.Translate(
            vabsdiffSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX VABSDIFF",
            vabsdiffPtx.Contains("sub.s32") &&
            vabsdiffPtx.Contains("abs.s32") &&
            vabsdiffPtx.Contains("add.s32 %r5"));
        var vabsdiff4Section = new rtxdisam.SassSection();
        vabsdiff4Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "VABSDIFF4.U8.ACC",
            Operands = { "R5", "R0", "R5", "R4" },
        });
        string vabsdiff4Ptx = rtxdisam.SassToPtx.Translate(
            vabsdiff4Section, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX VABSDIFF4",
            vabsdiff4Ptx.Split("bfe.u32").Length == 9 &&
            vabsdiff4Ptx.Contains("add.u32") &&
            vabsdiff4Ptx.Contains("mov.b32 %r5"));
        var prmtSection = new rtxdisam.SassSection();
        prmtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PRMT",
            Operands = { "R6", "R0", "0x5410", "R6" },
        });
        string prmtPtx = rtxdisam.SassToPtx.Translate(
            prmtSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX PRMT",
            prmtPtx.Contains("prmt.b32 %r6, %r0, %r6, 0x5410;"));
        var psetpSection = new rtxdisam.SassSection();
        psetpSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PSETP.AND.AND",
            Operands = { "P0", "PT", "!PT", "PT", "PT" },
        });
        string psetpPtx = rtxdisam.SassToPtx.Translate(
            psetpSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX PSETP",
            psetpPtx.Contains("and.b32 %rt0, 0, 1;") &&
            psetpPtx.Contains("and.b32 %rt1, %rt0, 1;") &&
            psetpPtx.Contains("setp.ne.u32 %p0, %rt1, 0;"));
        var isubLiftSection = new rtxdisam.SassSection();
        isubLiftSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISUB",
            Operands = { "R107", "0x5f", "R75" },
        });
        string isubPtx = rtxdisam.SassToPtx.Translate(
            isubLiftSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISUB",
            isubPtx.Contains("sub.s32 %r107, 0x5f, %r75;"));
        var setAliasSection = new rtxdisam.SassSection();
        setAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SET.LT.U32.S32",
            Operands = { "R1", "R2", "R3" },
        });
        setAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SETP.LT.AND.U32",
            Operands = { "P0", "R2", "R3", "P1" },
        });
        setAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SETP.NEU.AND.F32",
            Operands = { "P2", "R2", "R3", "P1" },
        });
        setAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SELP.F32",
            Operands = { "R4", "R2", "R3", "P0" },
        });
        setAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SLCT.S32.S32",
            Operands = { "R5", "R2", "R3", "R4" },
        });
        string setAliasPtx = rtxdisam.SassToPtx.Translate(
            setAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SET/SETP/SELP/SLCT",
            setAliasPtx.Contains("set.lt.u32.s32") &&
            setAliasPtx.Contains("setp.lt.and.u32") &&
            setAliasPtx.Contains("setp.neu.and.f32") &&
            setAliasPtx.Contains("selp.f32") &&
            setAliasPtx.Contains("slct.s32.s32"));
        var setNegatedPredSection = new rtxdisam.SassSection();
        setNegatedPredSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SETP.LT.AND.U32",
            Operands = { "P0", "R2", "R3", "!P1" },
        });
        setNegatedPredSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SELP.F32",
            Operands = { "R4", "R2", "R3", "!P0" },
        });
        string setNegatedPredPtx = rtxdisam.SassToPtx.Translate(
            setNegatedPredSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX SETP/SELP negated source predicate",
            setNegatedPredPtx.Contains("not.pred %pt0, %p1;") &&
            setNegatedPredPtx.Contains(
                "setp.lt.and.u32 %p0, %r2, %r3, %pt0;") &&
            setNegatedPredPtx.Contains("not.pred %pt1, %p0;") &&
            setNegatedPredPtx.Contains(
                "selp.f32 %r4, %r2, %r3, %pt1;"));
        var p2rSection = new rtxdisam.SassSection();
        p2rSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "P2R",
            Operands = { "R1", "P0" },
        });
        string p2rPtx = rtxdisam.SassToPtx.Translate(
            p2rSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX P2R simple",
            p2rPtx.Contains("selp.b32 %r1, 1, 0, %p0;"));
        var p2rPackSection = new rtxdisam.SassSection();
        p2rPackSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "P2R",
            Operands = { "R1", "PR" },
        });
        string p2rPackPtx = rtxdisam.SassToPtx.Translate(
            p2rPackSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_86");
        Check("SASS to PTX P2R simple PR-bank pack",
            p2rPackPtx.Contains("mov.b32 %r1, %rt") &&
            p2rPackPtx.Contains("selp.b32") &&
            p2rPackPtx.Contains("or.b32"));
        var p2rMaskedSection = new rtxdisam.SassSection();
        p2rMaskedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "P2R.B1",
            Operands = { "R1", "PR", "R2", "0x18" },
        });
        string p2rMaskedPtx = rtxdisam.SassToPtx.Translate(
            p2rMaskedSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX P2R masked insert",
            p2rMaskedPtx.Contains("selp.b32") &&
            p2rMaskedPtx.Contains("shl.b32") &&
            p2rMaskedPtx.Contains("and.b32") &&
            p2rMaskedPtx.Contains("or.b32 %r1"));
        var bitwiseSection = new rtxdisam.SassSection();
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "AND",
            Operands = { "R1", "R2", "R3" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "OR",
            Operands = { "R4", "R2", "R3" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "XOR",
            Operands = { "R5", "R2", "R3" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOT",
            Operands = { "R6", "R2" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CNOT",
            Operands = { "R7", "R2" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL.B64",
            Operands = { "R8", "R2", "R4" },
        });
        bitwiseSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHR.B64",
            Operands = { "R10", "R2", "R4" },
        });
        string bitwisePtx = rtxdisam.SassToPtx.Translate(
            bitwiseSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        var movShflPrmtSection = new rtxdisam.SassSection();
        movShflPrmtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV.B64",
            Operands = { "R1", "R2" },
        });
        movShflPrmtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV.PRED",
            Operands = { "P0", "P1" },
        });
        movShflPrmtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHFL.SYNC.BFLY.B32",
            Operands = { "R4", "R5", "R6", "R7", "R8" },
        });
        movShflPrmtSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PRMT.B32",
            Operands = { "R9", "R10", "R11", "R12" },
        });
        string movShflPrmtPtx = rtxdisam.SassToPtx.Translate(
            movShflPrmtSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        var memoryAliasSection = new rtxdisam.SassSection();
        memoryAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDU.U32",
            Operands = { "R1", "[R2]" },
        });
        memoryAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.ASYNC.SHARED.U32",
            Operands = { "[RZ]", "R4" },
        });
        memoryAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ST.BULK.SHARED",
            Operands = { "[RZ]", "16", "0" },
        });
        string memoryAliasPtx = rtxdisam.SassToPtx.Translate(
            memoryAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX LDU/ST.ASYNC/ST.BULK aliases",
            memoryAliasPtx.Contains("ldu.global.u32 %r1, [%rdt0+0];") &&
            memoryAliasPtx.Contains("st.shared.u32 [__shared], %r4;") &&
            memoryAliasPtx.Contains("st.shared.u64 [%rdt") &&
            memoryAliasPtx.Contains("sub.u32 %ru"));
        var extendedAliasSection = new rtxdisam.SassSection();
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.LD_REDUCE.ADD.U32",
            Operands = { "R1", "[R2]" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.ST.GLOBAL.U32",
            Operands = { "[R3]", "R4" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.RED.ADD.U32",
            Operands = { "[R5]", "R6" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.ST.ASYNC.RELEASE.GPU.GLOBAL.U32",
            Operands = { "[R7]", "R8" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PREFETCH.GLOBAL.L1",
            Operands = { "[R9]" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "PREFETCHU.L1",
            Operands = { "[R10]" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISSPACEP.GLOBAL",
            Operands = { "P0", "R11" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CVTA.TO.GLOBAL.U64",
            Operands = { "R12", "R13" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CVT.PACK.SAT.U8.S32.B32",
            Operands = { "R14", "R15", "R16", "R17" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CVT.FTZ.F64.F32",
            Operands = { "R22", "R24" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.CA.SHARED.GLOBAL",
            Operands = { "[R20]", "[R21]", "16" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.COMMIT_GROUP",
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.WAIT_GROUP",
            Operands = { "0" },
        });
        extendedAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.WAIT_ALL",
        });
        string extendedAliasPtx = rtxdisam.SassToPtx.Translate(
            extendedAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX multimem/prefetch/isspacep/cvta/cvt.pack/cp.async",
            extendedAliasPtx.Contains("ld.global.b32 %r1") &&
            extendedAliasPtx.Contains("st.global.b32") &&
            extendedAliasPtx.Contains("red.global.add.u32") &&
            extendedAliasPtx.Contains("st.release.gpu.global.b32") &&
            extendedAliasPtx.Contains("prefetch.global.L1") &&
            extendedAliasPtx.Contains("prefetchu.L1") &&
            extendedAliasPtx.Contains("isspacep.global %p0, %r11;") &&
            extendedAliasPtx.Contains("cvta.to.global.u64") &&
            extendedAliasPtx.Contains("cvt.pack.sat.u8.s32.b32 %r14, %r15, %r16, %r17;") &&
            extendedAliasPtx.Contains("cvt.ftz.f64.f32") &&
            extendedAliasPtx.Contains(".shared .align 16 .b8 __shared") &&
            extendedAliasPtx.Contains("cp.async.ca.shared.global") &&
            extendedAliasPtx.Contains("cp.async.commit_group;") &&
            extendedAliasPtx.Contains("cp.async.wait_group 0;") &&
            extendedAliasPtx.Contains("cp.async.wait_all;"));
        var bulkAliasSection = new rtxdisam.SassSection();
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.SHARED::CTA.GLOBAL.MBARRIER::COMPLETE_TX::BYTES",
            Operands = { "[R1]", "[R2]", "16", "[R3]" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.REDUCE.ASYNC.BULK.GLOBAL.SHARED::CTA.BULK_GROUP.ADD.U32",
            Operands = { "[R4]", "[R5]", "16" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.PREFETCH.L2.GLOBAL",
            Operands = { "[R6]", "16" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.CP.ASYNC.BULK.GLOBAL.SHARED::CTA.BULK_GROUP",
            Operands = { "[R7]", "[R8]", "16" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MULTIMEM.CP.REDUCE.ASYNC.BULK.GLOBAL.SHARED::CTA.BULK_GROUP.ADD.U32",
            Operands = { "[R9]", "[R10]", "16" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.TENSOR.1D.SHARED::CTA.GLOBAL.MBARRIER::COMPLETE_TX::BYTES",
            Operands = { "[R11]", "[R12, {R13}]", "[R14]" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.REDUCE.ASYNC.BULK.TENSOR.1D.GLOBAL.SHARED::CTA.ADD.BULK_GROUP",
            Operands = { "[R15, {R16}]", "[R17]" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.PREFETCH.TENSOR.1D.L2.GLOBAL",
            Operands = { "[R18, {R19}]" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.COMMIT_GROUP",
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CP.ASYNC.BULK.WAIT_GROUP",
            Operands = { "0" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "TENSORMAP.REPLACE.TILE.GLOBAL_ADDRESS.GLOBAL.B1024.B64",
            Operands = { "[R20]", "R21" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SPCOMPRESS.B8.B2.SP::2:4.X1",
            Operands = { "{R22}", "{R23}", "{R24, R25}", "R26" },
        });
        bulkAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SPDECOMPRESS.B8.B2.SP::2:4.X1",
            Operands = { "{R27}", "{R28}", "{R29}" },
        });
        string bulkAliasPtx = rtxdisam.SassToPtx.Translate(
            bulkAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_90");
        Check("SASS to PTX bulk/tensor/sparse aliases",
            bulkAliasPtx.Contains(".version 9.4", StringComparison.Ordinal) &&
            bulkAliasPtx.Contains("cp.async.bulk.shared::cta.global.") &&
            bulkAliasPtx.Contains("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32") &&
            bulkAliasPtx.Contains("cp.async.bulk.prefetch.L2.global") &&
            bulkAliasPtx.Contains("multimem.cp.async.bulk.global.shared::cta.bulk_group") &&
            bulkAliasPtx.Contains("multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32") &&
            bulkAliasPtx.Contains("cp.async.bulk.tensor.1d.shared::cta.global.") &&
            bulkAliasPtx.Contains("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.bulk_group") &&
            bulkAliasPtx.Contains("cp.async.bulk.prefetch.tensor.1d.L2.global") &&
            bulkAliasPtx.Contains("cp.async.bulk.commit_group;") &&
            bulkAliasPtx.Contains("cp.async.bulk.wait_group 0;") &&
            bulkAliasPtx.Contains("tensormap.replace.tile.global_address.global.b1024.b64") &&
            bulkAliasPtx.Contains("spcompress.b8.b2.sp::2:4.x1") &&
            bulkAliasPtx.Contains("spdecompress.b8.b2.sp::2:4.x1"));
        var surfaceAliasSection = new rtxdisam.SassSection();
        surfaceAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SULD.D.BA.2D.128.STRONG.SM.TRAP",
            Operands = { "R4", "[R4]", "0x0", "0x5a" },
        });
        surfaceAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SUST.D.BA.2D.STRONG.SM.TRAP",
            Operands = { "[R0]", "R4", "0x0", "0x5a" },
        });
        surfaceAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SURED.D.BA.2D.ADD.STRONG.SYS.TRAP",
            Operands = { "[R0]", "R4", "0x0", "0x5a" },
        });
        surfaceAliasSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SUQ.WIDTH.B32",
            Operands = { "R1", "[R2]" },
        });
        string surfaceAliasPtx = rtxdisam.SassToPtx.Translate(
            surfaceAliasSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX surface aliases",
            surfaceAliasPtx.Contains(".global .surfref __surf_0_90;") &&
            surfaceAliasPtx.Contains("suld.b.2d.v4.b32.trap") &&
            surfaceAliasPtx.Contains("sust.b.2d.b32.trap") &&
            surfaceAliasPtx.Contains("sured.b.add.2d.u32.trap") &&
            surfaceAliasPtx.Contains("suq.width.b32"));
        Check("SASS to PTX MOV/SHFL.SYNC/PRMT aliases",
            movShflPrmtPtx.Contains("mov.b64") &&
            movShflPrmtPtx.Contains("mov.pred %p0, %p1;") &&
            movShflPrmtPtx.Contains("shfl.sync.bfly.b32 %r4, %r5, %r6, %r7, %r8;") &&
            movShflPrmtPtx.Contains("prmt.b32 %r9, %r10, %r11, %r12;"));
        Check("SASS to PTX bitwise aliases",
            bitwisePtx.Contains("and.b32") &&
            bitwisePtx.Contains("or.b32") &&
            bitwisePtx.Contains("xor.b32") &&
            bitwisePtx.Contains("not.b32") &&
            bitwisePtx.Contains("cnot.b32") &&
            bitwisePtx.Contains("shl.b64") &&
            bitwisePtx.Contains("shr.b64"));
        var i2iSection = new rtxdisam.SassSection();
        i2iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2I.U8.S32",
            Operands = { "R1", "R2" },
        });
        i2iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2I.S16.S32.SAT",
            Operands = { "R3", "R4" },
        });
        i2iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "I2I.U32.U32",
            Operands = { "R5", "R6" },
        });
        string i2iPtx = rtxdisam.SassToPtx.Translate(
            i2iSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX I2I widths",
            i2iPtx.Contains("cvt.u8.s32 %r1, %r2;") &&
            i2iPtx.Contains("cvt.sat.s16.s32 %r3, %r4;") &&
            i2iPtx.Contains("mov.b32 %r5, %r6;"));
        var isetSection = new rtxdisam.SassSection();
        isetSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISET.GE.U32.AND",
            Operands = { "R2", "R6", "0x65", "PT" },
        });
        string isetPtx = rtxdisam.SassToPtx.Translate(
            isetSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX ISET",
            isetPtx.Contains("setp.ge.u32 %pt0, %r6, 0x65;") &&
            isetPtx.Contains("selp.b32 %r2, 1, 0, %pt0;"));
        var fsetSection = new rtxdisam.SassSection();
        fsetSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FSET.GEU.AND",
            Operands = { "R6", "R2", "1", "PT" },
        });
        string fsetPtx = rtxdisam.SassToPtx.Translate(
            fsetSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FSET",
            fsetPtx.Contains("setp.geu.f32 %pt0, %r2, 1.0;") &&
            fsetPtx.Contains("selp.b32 %r6, 1, 0, %pt0;"));
        var fsetpSection = new rtxdisam.SassSection();
        fsetpSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FSETP.GEU.AND",
            Operands = { "P1", "PT", "|R2|", "1.175494350822287508e-38", "PT" },
        });
        string fsetpPtx = rtxdisam.SassToPtx.Translate(
            fsetpSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FSETP",
            fsetpPtx.Contains("abs.f32 %ft0, %r2;") &&
            fsetpPtx.Contains(
                "setp.geu.f32 %p1, %ft0, 1.175494350822287508e-38;"));
        string syncSassPath = Path.GetTempFileName();
        try
        {
            File.WriteAllText(syncSassPath,
                ".target sm_50\n" +
                ".section .text.k,\"ax\",@progbits\n" +
                "@P0 SYNC (*\"TARGET= .L_1 \"*);\n" +
                ".L_1:\n" +
                "EXIT;\n");
            var syncProgram = rtxdisam.SassParser.Parse(syncSassPath);
            var syncInstruction = syncProgram.Sections
                .SelectMany(section => section.Nodes)
                .OfType<rtxdisam.SassInstruction>()
                .First(instruction => instruction.Mnemonic.StartsWith("SYNC"));
            Check("SASS parser SYNC target annotation",
                syncInstruction.BranchTargets.Contains(".L_1"));
        }
        finally
        {
            File.Delete(syncSassPath);
        }
        var syncSection = new rtxdisam.SassSection();
        syncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SSY",
            Operands = { "`(.L_1)`" },
        });
        syncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SYNC",
            Guard = "@P0",
            BranchTargets = { ".L_1" },
        });
        syncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "R3" },
        });
        syncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SYNC",
            BranchTargets = { ".L_1" },
        });
        syncSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        syncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string syncPtx = rtxdisam.SassToPtx.Translate(
            syncSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX SYNC reconvergence",
            Regex.IsMatch(syncPtx, @"@%p0 bra L_\d+;") &&
            !syncPtx.Contains("SSY"));
        Check("SASS to PTX SYNC to next label",
            !Regex.IsMatch(syncPtx, @"(?m)^bra L_\d+;"));
        var earlyExitSection = new rtxdisam.SassSection();
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SSY",
            Operands = { "`(.L_1)`" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "c[0x0][0x20]" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "S2R",
            Operands = { "R0", "SR_TID.X" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.NE.U32.AND",
            Operands = { "P0", "PT", "R0", "RZ", "PT" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SYNC",
            Guard = "@P0",
            BranchTargets = { ".L_1" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R2", "[R2]" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ATOM.E.MAX",
            Operands = { "R6", "[R4]", "R2" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SYNC",
            BranchTargets = { ".L_1" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
            Guard = "@P0",
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4+0x4]", "R6" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        earlyExitSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_2" });
        earlyExitSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_2)`" },
        });
        earlyExitSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_21" });
        string earlyExitPtx = rtxdisam.SassToPtx.Translate(
            earlyExitSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX early-exit reconvergence",
            earlyExitPtx.Contains("mov.u32 %r6, 0;") &&
            Regex.IsMatch(earlyExitPtx, @"@%p0 bra L_\d+;") &&
            earlyExitPtx.Contains("atom.global.max.u32 %r6, [") &&
            earlyExitPtx.Contains("st.global.u32 [") &&
            earlyExitPtx.Contains("], %r6;") &&
            !earlyExitPtx.Contains("@!%p0 st.global") &&
            Regex.Matches(earlyExitPtx, @"(?m)^L_\d+:").Count == 1 &&
            Regex.IsMatch(earlyExitPtx, @"(?m)^L_\d+:\nret;") &&
            !Regex.IsMatch(earlyExitPtx, @"(?m)^bra L_\d+;"));
        var cctlSection = new rtxdisam.SassSection();
        cctlSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CCTL.E.IV",
            Operands = { "[R2+0x28]" },
        });
        string cctlPtx = rtxdisam.SassToPtx.Translate(
            cctlSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX CCTL.E.IV", cctlPtx.Contains("membar.gl;"));
        var cctlAllSection = new rtxdisam.SassSection();
        cctlAllSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CCTL.E.IVALL",
        });
        string cctlAllPtx = rtxdisam.SassToPtx.Translate(
            cctlAllSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX CCTL.E.IVALL", cctlAllPtx.Contains("membar.gl;"));
        var cctlUnsupportedSection = new rtxdisam.SassSection();
        cctlUnsupportedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CCTL.E.WB",
            Operands = { "[R2]" },
        });
        bool rejectedCctl = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                cctlUnsupportedSection, "k",
                Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedCctl = true;
        }
        Check("SASS to PTX rejects unsupported CCTL mode", rejectedCctl);
        var texdepbarSection = new rtxdisam.SassSection();
        texdepbarSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "TEXDEPBAR",
            Operands = { "0x0" },
        });
        string texdepbarPtx = rtxdisam.SassToPtx.Translate(
            texdepbarSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX TEXDEPBAR",
            !texdepbarPtx.Contains("TEXDEPBAR"));
        var bptSection = new rtxdisam.SassSection();
        bptSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BPT.TRAP",
            Operands = { "0x1" },
        });
        string bptPtx = rtxdisam.SassToPtx.Translate(
            bptSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX BPT", bptPtx.Contains("brkpt;"));
        var redundantBranchSection = new rtxdisam.SassSection();
        redundantBranchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_1)`" },
        });
        redundantBranchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        redundantBranchSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string redundantBranchPtx = rtxdisam.SassToPtx.Translate(
            redundantBranchSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX redundant BRA",
            !redundantBranchPtx.Contains("bra L__L_1;"));
        var switchSection = new rtxdisam.SassSection();
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.EQ.AND",
            Operands = { "P0", "PT", "R0", "RZ", "PT" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_1)`" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_2)`" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x1" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_3)`" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_2" });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x2" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_3)`" },
        });
        switchSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_3" });
        switchSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string switchPtx = rtxdisam.SassToPtx.Translate(
            switchSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX switch select",
            switchPtx.Contains("mov.u32 %rt1, 2;") &&
            switchPtx.Contains("setp.eq.u32 %pt0, %rt0, 0;") &&
            switchPtx.Contains("@%pt0 mov.u32 %rt1, 1;") &&
            switchPtx.Contains("st.global.u32 ["));
        var leafCallSection = new rtxdisam.SassSection();
        leafCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_1)`" },
        });
        leafCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        leafCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        leafCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x7" },
        });
        leafCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R2]", "R0" },
        });
        leafCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string leafCallPtx = rtxdisam.SassToPtx.Translate(
            leafCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX leaf CAL inline",
            leafCallPtx.Contains("mov.u32 %r0, 0x7;") &&
            leafCallPtx.Contains("st.global.u32") &&
            !leafCallPtx.Contains("CAL"));
        var callRelNoIncSection = new rtxdisam.SassSection();
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x7" },
        });
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CALL.REL.NOINC",
            Operands = { "`(.L_callrel)`" },
        });
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_callrel" });
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R2", "0x9" },
        });
        callRelNoIncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string callRelNoIncPtx = rtxdisam.SassToPtx.Translate(
            callRelNoIncSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_80");
        Check("SASS to PTX CALL.REL.NOINC inline",
            callRelNoIncPtx.Contains("mov.u32 %r0, 0x7;") &&
            callRelNoIncPtx.Contains("mov.u32 %r2, 0x9;") &&
            !callRelNoIncPtx.Contains("CALL") &&
            !callRelNoIncPtx.Contains("CAL"));
        var paramSlotSection = new rtxdisam.SassSection();
        paramSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STL",
            Operands = { "[R1]", "R2" },
        });
        paramSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_empty)`" },
        });
        paramSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDL",
            Operands = { "R3", "[R1+0x4]" },
        });
        paramSlotSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        paramSlotSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_empty" });
        paramSlotSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string paramSlotPtx = rtxdisam.SassToPtx.Translate(
            paramSlotSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX caller parameter slots",
            paramSlotPtx.Contains(
                "st.local.u32 [__callee_frame+0], %r2;") &&
            paramSlotPtx.Contains(
                "ld.local.u32 %r3, [__callee_frame+0x4];"));
        var movedSlotSection = new rtxdisam.SassSection();
        movedSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STL",
            Operands = { "[R1]", "R2" },
        });
        movedSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R1", "R1", "0x4" },
        });
        movedSlotSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDL",
            Operands = { "R3", "[R1]" },
        });
        bool rejectedMovedSlot = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                movedSlotSection, "k",
                Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedMovedSlot = true;
        }
        Check("SASS to PTX rejects moved caller frame", rejectedMovedSlot);
        var nestedCallSection = new rtxdisam.SassSection();
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        nestedCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_g)`" },
        });
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        nestedCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_g" });
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x1" },
        });
        nestedCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string nestedCallPtx = rtxdisam.SassToPtx.Translate(
            nestedCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX nested leaf CAL inline",
            !nestedCallPtx.Contains("CAL") &&
            nestedCallPtx.Contains("L_inline_0_") &&
            nestedCallPtx.Contains("L_inline_1_") &&
            nestedCallPtx.Contains("add.s32 %r0, %r0, 0x1;"));
        var recursiveCallSection = new rtxdisam.SassSection();
        recursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        recursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        recursiveCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        recursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        recursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string recursivePtx = rtxdisam.SassToPtx.Translate(
            recursiveCallSection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX tail-recursive CAL",
            recursivePtx.Contains("bra L_inline_", StringComparison.Ordinal) &&
            !recursivePtx.Contains("CAL", StringComparison.Ordinal));
        var functionCallSection = new rtxdisam.SassSection();
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R0" },
        });
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        functionCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R0" },
        });
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x1" },
        });
        functionCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string functionCallPtx = rtxdisam.SassToPtx.Translate(
            functionCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50",
            emitFunctions: true);
        Check("SASS to PTX real function call",
            functionCallPtx.Contains(".func (.param .u32 __ret)") &&
            functionCallPtx.Contains("call.uni (__fc0_r)") &&
            functionCallPtx.Contains("st.param.u32 [__ret], %r0;") &&
            !functionCallPtx.Contains("L_inline_") &&
            !functionCallPtx.Contains("CAL"));
        var multiReturnFuncSection = new rtxdisam.SassSection();
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R0" },
        });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "R2" },
        });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R0" },
        });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "R2" },
        });
        multiReturnFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string multiReturnFuncPtx = rtxdisam.SassToPtx.Translate(
            multiReturnFuncSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", emitFunctions: true);
        Check("SASS to PTX packs two function returns",
            multiReturnFuncPtx.Contains(".func (.param .u64 __ret)") &&
            multiReturnFuncPtx.Contains("mov.b64 %rdt") &&
            !multiReturnFuncPtx.Contains("L_inline_"));
        var pointerFuncSection = new rtxdisam.SassSection();
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R0" },
        });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDG.E",
            Operands = { "R0", "[R2]" },
        });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R0" },
        });
        pointerFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string pointerFuncPtx = rtxdisam.SassToPtx.Translate(
            pointerFuncSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", emitFunctions: true);
        Check("SASS to PTX function pointer argument",
            Regex.IsMatch(pointerFuncPtx,
                @"\.func \(\.param \.u32 __ret\) fn_L_\d+\(\.param \.u64 __a0\)") &&
            pointerFuncPtx.Contains("ld.param.u64 %rdfn0, [__a0];") &&
            pointerFuncPtx.Contains("ld.global.u32 %r0, [%rdfn0+0];") &&
            !pointerFuncPtx.Contains("L_inline_"));
        var localFuncSection = new rtxdisam.SassSection();
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_f)`" },
        });
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "R0" },
        });
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        localFuncSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_f" });
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STL",
            Operands = { "[R1]", "R0" },
        });
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDL",
            Operands = { "R0", "[R1]" },
        });
        localFuncSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string localFuncPtx = rtxdisam.SassToPtx.Translate(
            localFuncSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", emitFunctions: true);
        Check("SASS to PTX function local frame",
            localFuncPtx.Contains(".local .align 16 .b8 __callee_frame[16];") &&
            localFuncPtx.Contains("st.local.u32 [__callee_frame+0], %r0;") &&
            localFuncPtx.Contains("ld.local.u32 %r0, [__callee_frame+0];") &&
            !localFuncPtx.Contains("L_inline_"));
        var loopsCallSection = new rtxdisam.SassSection();
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        loopsCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "R0" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R5", "RZ" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LE.U32.AND",
            Operands = { "P0", "PT", "R4", "RZ", "PT" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_done)`" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_loop" });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R5", "R5", "R4" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R4", "R4", "-0x1" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.GT.U32.AND",
            Operands = { "P0", "PT", "R4", "RZ", "PT" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_loop)`" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_done" });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R5" },
        });
        loopsCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string loopsCallPtx = rtxdisam.SassToPtx.Translate(
            loopsCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX branchy leaf CAL inline",
            loopsCallPtx.Contains("L_inline_0__L_loop:") &&
            loopsCallPtx.Contains("bra L_inline_0__L_done;") &&
            loopsCallPtx.Contains("bra L_inline_0__L_loop;") &&
            !loopsCallPtx.Contains("CAL"));
        var earlyReturnCallSection = new rtxdisam.SassSection();
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0xa", "PT" },
        });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_small)`" },
        });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x64" },
        });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_small" });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x1" },
        });
        earlyReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string earlyReturnCallPtx = rtxdisam.SassToPtx.Translate(
            earlyReturnCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX early-return CAL if-conversion",
            earlyReturnCallPtx.Contains("@!%p0 add.s32 %r0, %r0, 0x64;") &&
            earlyReturnCallPtx.Contains("@%p0 add.s32 %r0, %r0, 0x1;") &&
            !earlyReturnCallPtx.Contains("L_inline_") &&
            !earlyReturnCallPtx.Contains("CAL"));
        var singleMergeCallSection = new rtxdisam.SassSection();
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0xa", "PT" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_then)`" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x64" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_merge)`" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_then" });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x1" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_merge" });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "R0" },
        });
        singleMergeCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string singleMergeCallPtx = rtxdisam.SassToPtx.Translate(
            singleMergeCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX single-RET diamond CAL if-conversion",
            singleMergeCallPtx.Contains("@!%p0 add.s32 %r0, %r0, 0x64;") &&
            singleMergeCallPtx.Contains("@%p0 add.s32 %r0, %r0, 0x1;") &&
            !singleMergeCallPtx.Contains("bra ") &&
            !singleMergeCallPtx.Contains("CAL"));
        var guardedReturnCallSection = new rtxdisam.SassSection();
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0xa", "PT" },
        });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "RET",
            Guard = "@P0",
        });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "0x64" },
        });
        guardedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string guardedReturnCallPtx = rtxdisam.SassToPtx.Translate(
            guardedReturnCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX guarded-RET CAL if-conversion",
            guardedReturnCallPtx.Contains("@!%p0 add.s32 %r0, %r0, 0x64;") &&
            !guardedReturnCallPtx.Contains("CAL"));
        var nestedReturnCallSection = new rtxdisam.SassSection();
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Guard = "@P0",
            Operands = { "`(.L_then)`" },
        });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_then" });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_other)`" },
        });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_other" });
        nestedReturnCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string nestedReturnCallPtx = rtxdisam.SassToPtx.Translate(
            nestedReturnCallSection, "k",
            Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX non-diamond multi-return CAL",
            nestedReturnCallPtx.Contains("L_inline_0_exit:") &&
            nestedReturnCallPtx.Contains("bra L_inline_0_exit;") &&
            !nestedReturnCallPtx.Contains("CAL"));
        var branchyCallSection = new rtxdisam.SassSection();
        branchyCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "`(.L_func)`" },
        });
        branchyCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        branchyCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_func" });
        branchyCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_outside)`" },
        });
        branchyCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        branchyCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_outside" });
        branchyCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        bool rejectedBranchyCall = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                branchyCallSection, "k",
                Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedBranchyCall = true;
        }
        Check("SASS to PTX rejects CAL branch escaping callee", rejectedBranchyCall);
        var separateEntrySection = new rtxdisam.SassSection
        {
            Name = ".text.k",
            Architecture = "sm_50",
        };
        separateEntrySection.Nodes.Add(new rtxdisam.SassLabel { Name = "k" });
        separateEntrySection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "f" },
        });
        separateEntrySection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        var separateCalleeSection = new rtxdisam.SassSection
        {
            Name = ".text.f",
            Architecture = "sm_50",
        };
        separateCalleeSection.Nodes.Add(new rtxdisam.SassLabel { Name = "f" });
        separateCalleeSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x7" },
        });
        separateCalleeSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string separateSectionPtx = rtxdisam.SassToPtx.Translate(
            separateEntrySection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50", linkedSections: new[]
            {
                separateEntrySection,
                separateCalleeSection,
            });
        Check("SASS to PTX inlines a CAL in a separate .text section",
            separateSectionPtx.Contains("mov.u32 %r0, 0x7;",
                StringComparison.Ordinal) &&
            !separateSectionPtx.Contains("CAL", StringComparison.Ordinal));
        var jcalSection = new rtxdisam.SassSection();
        var jcalCall = new rtxdisam.SassInstruction
        {
            Mnemonic = "JCAL",
            Operands = { "0x0" },
        };
        jcalCall.Relocations.Add(new rtxdisam.SassRelocation
        {
            Symbol = "f",
            Type = 0x4b,
        });
        jcalSection.Nodes.Add(jcalCall);
        jcalSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        jcalSection.Nodes.Add(new rtxdisam.SassLabel { Name = "f" });
        jcalSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0xb" },
        });
        jcalSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string jcalPtx = rtxdisam.SassToPtx.Translate(
            jcalSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX inlines relocatable JCAL",
            jcalPtx.Contains("mov.u32 %r0, 0xb;",
                StringComparison.Ordinal) &&
            !jcalPtx.Contains("JCAL", StringComparison.Ordinal));
        var dynamicFrameCallSection = new rtxdisam.SassSection();
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R2", "0x10" },
        });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "f" },
        });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = "f" });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R1", "R1", "-R2" },
        });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STL",
            Operands = { "[R1]", "R0" },
        });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDL",
            Operands = { "R3", "[R1]" },
        });
        dynamicFrameCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string dynamicFramePtx = rtxdisam.SassToPtx.Translate(
            dynamicFrameCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50");
        Check("SASS to PTX dynamic CAL stack growth",
            dynamicFramePtx.Contains("__callee_dynamic_frame[65536];",
                StringComparison.Ordinal) &&
            dynamicFramePtx.Contains("sub.u64 %rdt", StringComparison.Ordinal) &&
            dynamicFramePtx.Contains("st.local.u32 [%rdt", StringComparison.Ordinal) &&
            dynamicFramePtx.Contains("ld.local.u32 %r3, [%rdt",
                StringComparison.Ordinal));
        var indirectCallSection = new rtxdisam.SassSection();
        indirectCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R2", "0x20" },
        });
        indirectCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "R2" },
        });
        indirectCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        indirectCallSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = "f",
            Address = 0x20,
        });
        indirectCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x9" },
        });
        indirectCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string indirectCallPtx = rtxdisam.SassToPtx.Translate(
            indirectCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50");
        Check("SASS to PTX constant-indirect CAL",
            indirectCallPtx.Contains("mov.u32 %r0, 0x9;",
                StringComparison.Ordinal) &&
            !indirectCallPtx.Contains("CAL", StringComparison.Ordinal));
        var tailRecursiveCallSection = new rtxdisam.SassSection();
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x3" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "f" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = "f" });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.NE.U32.AND",
            Operands = { "P0", "PT", "R0", "RZ", "PT" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_recur)`" },
            Guard = "@P0",
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R2", "0x7" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "BRA",
            Operands = { "`(.L_done)`" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_recur" });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD32I",
            Operands = { "R0", "R0", "-0x1" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "f" },
        });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_done" });
        tailRecursiveCallSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string tailRecursivePtx = rtxdisam.SassToPtx.Translate(
            tailRecursiveCallSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50");
        Check("SASS to PTX tail-recursive CAL with base case",
            tailRecursivePtx.Contains("bra L_inline_", StringComparison.Ordinal) &&
            !tailRecursivePtx.Contains("CAL", StringComparison.Ordinal));
        var mutualTailSection = new rtxdisam.SassSection();
        void AddMutual(string mnemonic, params string[] operands)
        {
            var instruction = new rtxdisam.SassInstruction { Mnemonic = mnemonic };
            instruction.Operands.AddRange(operands);
            mutualTailSection.Nodes.Add(instruction);
        }
        void AddMutualGuarded(string mnemonic, params string[] operands)
        {
            var instruction = new rtxdisam.SassInstruction
            {
                Mnemonic = mnemonic,
                Guard = "@P0",
            };
            instruction.Operands.AddRange(operands);
            mutualTailSection.Nodes.Add(instruction);
        }
        AddMutual("MOV32I", "R0", "0x2");
        AddMutual("CAL", "A");
        AddMutual("EXIT");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = "A" });
        AddMutual("ISETP.NE.U32.AND", "P0", "PT", "R0", "RZ", "PT");
        AddMutualGuarded("BRA", "`(.L_a_recur)`");
        AddMutual("MOV32I", "R2", "0x7");
        AddMutual("BRA", "`(.L_a_done)`");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_a_recur" });
        AddMutual("IADD32I", "R0", "R0", "-0x1");
        AddMutual("CAL", "B");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_a_done" });
        AddMutual("RET");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = "B" });
        AddMutual("ISETP.NE.U32.AND", "P0", "PT", "R0", "RZ", "PT");
        AddMutualGuarded("BRA", "`(.L_b_recur)`");
        AddMutual("MOV32I", "R2", "0x9");
        AddMutual("BRA", "`(.L_b_done)`");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_b_recur" });
        AddMutual("IADD32I", "R0", "R0", "-0x1");
        AddMutual("CAL", "A");
        mutualTailSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_b_done" });
        AddMutual("RET");
        string mutualTailPtx = rtxdisam.SassToPtx.Translate(
            mutualTailSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50");
        Check("SASS to PTX mutual tail recursion",
            mutualTailPtx.Contains("bra L_inline_", StringComparison.Ordinal) &&
            !mutualTailPtx.Contains("CAL", StringComparison.Ordinal));
        var runtimePointerSection = new rtxdisam.SassSection();
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "CAL",
            Operands = { "R2" },
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = "f",
            Address = 0x20,
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x7" },
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassLabel
        {
            Name = "g",
            Address = 0x30,
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x9" },
        });
        runtimePointerSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "RET" });
        string runtimePointerPtx = rtxdisam.SassToPtx.Translate(
            runtimePointerSection, "k", Array.Empty<rtxdisam.PtxParameter>(),
            "sm_50");
        Check("SASS to PTX runtime function-pointer dispatch",
            runtimePointerPtx.Contains(
                "setp.eq.u32 %pt0, %r2, 0x20;", StringComparison.Ordinal) &&
            runtimePointerPtx.Contains(
                "setp.eq.u32 %pt0, %r2, 0x30;", StringComparison.Ordinal) &&
            runtimePointerPtx.Contains("mov.u32 %r0, 0x7;",
                StringComparison.Ordinal) &&
            runtimePointerPtx.Contains("mov.u32 %r0, 0x9;",
                StringComparison.Ordinal) &&
            !runtimePointerPtx.Contains("CAL", StringComparison.Ordinal));
        var dynamicLdcSection = new rtxdisam.SassSection();
        void AddDynamicLdc(string mnemonic, params string[] operands)
        {
            var dynamicInstruction = new rtxdisam.SassInstruction
            {
                Mnemonic = mnemonic,
            };
            dynamicInstruction.Operands.AddRange(operands);
            dynamicLdcSection.Nodes.Add(dynamicInstruction);
        }
        AddDynamicLdc("MOV", "R0", "c[0x0][0x140]");
        AddDynamicLdc("MOV32I", "R1", "0x148");
        AddDynamicLdc("IMAD", "R0", "R0", "0x8", "R1");
        AddDynamicLdc("ISETP.LT.U32.AND", "P0", "PT", "R0", "0x160", "PT");
        dynamicLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Guard = "@!P0",
            Operands = { "R0", "0x148" },
        });
        AddDynamicLdc("LDC.64", "R2", "c[0x0][R0]");
        AddDynamicLdc("MOV", "R4", "c[0x0][0x160]");
        AddDynamicLdc("STG.E", "[R4]", "R2");
        AddDynamicLdc("EXIT");
        var dynamicParameters = new[]
        {
            new rtxdisam.PtxParameter("p0", "u32", 0x140, 4),
            new rtxdisam.PtxParameter("p1", "u32", 0x148, 4),
            new rtxdisam.PtxParameter("p2", "u32", 0x14c, 4),
            new rtxdisam.PtxParameter("p3", "u32", 0x150, 4),
            new rtxdisam.PtxParameter("p4", "u32", 0x154, 4),
            new rtxdisam.PtxParameter("p5", "u32", 0x158, 4),
            new rtxdisam.PtxParameter("p6", "u32", 0x15c, 4),
            new rtxdisam.PtxParameter("p7", "u64", 0x160, 8),
        };
        string dynamicLdcPtx = rtxdisam.SassToPtx.Translate(
            dynamicLdcSection, "k", dynamicParameters, "sm_50");
        Check("SASS to PTX bounded dynamic LDC",
            dynamicLdcPtx.Contains("setp.ge.u32") &&
            dynamicLdcPtx.Contains("ld.shared.u32 %r2") &&
            dynamicLdcPtx.Contains("ld.shared.u32 %r3") &&
            dynamicLdcPtx.Contains("__shared") &&
            !dynamicLdcPtx.Contains("c[0x0][R"));
        var affineLdcSection = new rtxdisam.SassSection();
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "c[0x0][0x140]" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R1", "0x148" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0x3", "PT" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Guard = "@!P0",
            Operands = { "R0", "0x0" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R0", "R0", "0x8", "R1" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.64",
            Operands = { "R2", "c[0x0][R0]" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x160]" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R2" },
        });
        affineLdcSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string affineLdcPtx = rtxdisam.SassToPtx.Translate(
            affineLdcSection, "k", dynamicParameters, "sm_50");
        var boundedOffsetLdcSection = new rtxdisam.SassSection();
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "c[0x0][0x140]" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R1", "0x148" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0x3", "PT" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IMAD",
            Operands = { "R0", "R0", "0x8", "R1" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Guard = "@!P0",
            Operands = { "R0", "0x0" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.64",
            Operands = { "R2", "c[0x0][R0+0x8]" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x160]" },
        });
        boundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R2" },
        });
        boundedOffsetLdcSection.Nodes.Add(
            new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string boundedOffsetLdcPtx = rtxdisam.SassToPtx.Translate(
            boundedOffsetLdcSection, "k", dynamicParameters, "sm_50");
        Check("SASS to PTX bounded dynamic LDC displacement",
            boundedOffsetLdcPtx.Contains("add.s32 %rt", StringComparison.Ordinal) &&
            boundedOffsetLdcPtx.Contains(", 8;", StringComparison.Ordinal) &&
            boundedOffsetLdcPtx.Contains("ld.shared.u32 %r2") &&
            boundedOffsetLdcPtx.Contains("ld.shared.u32 %r3") &&
            !boundedOffsetLdcPtx.Contains("c[0x0][R"));
        Check("SASS to PTX affine bounded dynamic LDC",
            affineLdcPtx.Contains("mov.u32 %r1, 0x148;") &&
            affineLdcPtx.Contains("mad.lo.s32 %r0, %r0, 0x8, %r1;") &&
            affineLdcPtx.Contains("ld.shared.u32 %r2") &&
            affineLdcPtx.Contains("__shared") &&
            !affineLdcPtx.Contains("c[0x0][R"));
        var shlLdcSection = new rtxdisam.SassSection();
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R0", "c[0x0][0x140]" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R1", "0x148" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.LT.U32.AND",
            Operands = { "P0", "PT", "R0", "0x3", "PT" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Guard = "@!P0",
            Operands = { "R0", "0x0" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SHL",
            Operands = { "R0", "R0", "0x3" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R0", "R0", "R1" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.64",
            Operands = { "R2", "c[0x0][R0]" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R4", "c[0x0][0x160]" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STG.E",
            Operands = { "[R4]", "R2" },
        });
        shlLdcSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string shlLdcPtx = rtxdisam.SassToPtx.Translate(
            shlLdcSection, "k", dynamicParameters, "sm_50");
        Check("SASS to PTX SHL/IADD bounded dynamic LDC",
            shlLdcPtx.Contains("shl.b32 %r0, %r0, 0x3;") &&
            shlLdcPtx.Contains("add.s32 %r0, %r0, %r1;") &&
            shlLdcPtx.Contains("ld.shared.u32 %r2") &&
            shlLdcPtx.Contains("__shared") &&
            !shlLdcPtx.Contains("c[0x0][R"));
        var ldc128Section = new rtxdisam.SassSection();
        ldc128Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x140" },
        });
        ldc128Section.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.128",
            Operands = { "R2", "c[0x0][R0]" },
        });
        ldc128Section.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        bool rejectedLdc128 = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                ldc128Section, "k",
                new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
                "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedLdc128 = true;
        }
        Check("SASS to PTX rejects LDC.128", rejectedLdc128);
        var unboundedLdcSection = new rtxdisam.SassSection();
        unboundedLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        unboundedLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.64",
            Operands = { "R2", "c[0x0][R2]" },
        });
        unboundedLdcSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string unboundedLdcPtx = rtxdisam.SassToPtx.Translate(
            unboundedLdcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX supports unbounded dynamic LDC",
            unboundedLdcPtx.Contains(
                "ld.shared.u32 %r2, [", StringComparison.Ordinal) &&
            unboundedLdcPtx.Contains(
                "ld.shared.u32 %r3, [", StringComparison.Ordinal));
        var unboundedOffsetLdcSection = new rtxdisam.SassSection();
        unboundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "c[0x0][0x140]" },
        });
        unboundedOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.64",
            Operands = { "R2", "c[0x0][R2+0x8]" },
        });
        unboundedOffsetLdcSection.Nodes.Add(
            new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string unboundedOffsetLdcPtx = rtxdisam.SassToPtx.Translate(
            unboundedOffsetLdcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX supports unbounded dynamic LDC displacement",
            unboundedOffsetLdcPtx.Contains(
                "add.s32 %rt", StringComparison.Ordinal) &&
            unboundedOffsetLdcPtx.Contains(", 8;", StringComparison.Ordinal) &&
            unboundedOffsetLdcPtx.Contains(
                "ld.shared.u32 %r2, [", StringComparison.Ordinal));
        var constantOffsetLdcSection = new rtxdisam.SassSection();
        constantOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x140" },
        });
        constantOffsetLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC",
            Operands = { "R1", "c[0x0][R0+0x4]" },
        });
        constantOffsetLdcSection.Nodes.Add(
            new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string constantOffsetLdcPtx = rtxdisam.SassToPtx.Translate(
            constantOffsetLdcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u32", 0x144, 4) },
            "sm_50");
        Check("SASS to PTX folds a constant LDC address displacement",
            constantOffsetLdcPtx.Contains("mov.u32 %r1, %rpp0;",
                StringComparison.Ordinal));
        var jumpTableSection = new rtxdisam.SassSection();
        void AddJumpInstruction(string mnemonic, int address, params string[] operands)
        {
            var jumpInstruction = new rtxdisam.SassInstruction
            {
                Mnemonic = mnemonic,
                Address = address,
            };
            jumpInstruction.Operands.AddRange(operands);
            jumpTableSection.Nodes.Add(jumpInstruction);
        }
        AddJumpInstruction("MOV", 0x10, "R0", "c[0x0][0x140]");
        AddJumpInstruction("SHF.L", 0x18, "R0", "RZ", "0x2", "R0");
        AddJumpInstruction("LDC", 0x20, "R2", "c[0x2][R0]");
        AddJumpInstruction("BRX", 0x28, "R2", "-0x30");
        AddJumpInstruction("MOV", 0x30, "RZ", "RZ");
        AddJumpInstruction("EXIT", 0x38);
        AddJumpInstruction("MOV", 0x48, "RZ", "RZ");
        AddJumpInstruction("EXIT", 0x50);
        AddJumpInstruction("MOV", 0x58, "RZ", "RZ");
        AddJumpInstruction("EXIT", 0x60);
        uint[] jumpTableWords = { 0x30, 0x40, 0x58 };
        string jumpTablePtx = rtxdisam.SassToPtx.Translate(
            jumpTableSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u32", 0x140, 4) },
            "sm_35", null, 0,
            offset => offset >= 0 && offset % 4 == 0 &&
                      offset / 4 < jumpTableWords.Length
                ? jumpTableWords[offset / 4]
                : null,
            jumpTableWords.Length * 4);
        Check("SASS to PTX jump table",
            jumpTablePtx.Contains("setp.eq.u32 %pt0, %r0, 0;") &&
            jumpTablePtx.Contains("@%pt0 bra L_jt_0;") &&
            jumpTablePtx.Contains("setp.eq.u32 %pt1, %r0, 4;") &&
            jumpTablePtx.Contains("@%pt1 bra L_jt_1;") &&
            jumpTablePtx.Contains("setp.eq.u32 %pt2, %r0, 8;") &&
            jumpTablePtx.Contains("@%pt2 bra L_jt_2;") &&
            jumpTablePtx.Contains("L_jt_0:") &&
            jumpTablePtx.Contains("L_jt_1:") &&
            jumpTablePtx.Contains("L_jt_2:") &&
            jumpTablePtx.Contains("L_exit_common:") &&
            jumpTablePtx.Contains("ret;"));
        var constantLdcSection = new rtxdisam.SassSection();
        constantLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV32I",
            Operands = { "R0", "0x140" },
        });
        constantLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC",
            Operands = { "R0", "c[0x0][R0]" },
        });
        constantLdcSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string constantLdcPtx = rtxdisam.SassToPtx.Translate(
            constantLdcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u32", 0x140, 4) },
            "sm_50");
        Check("SASS to PTX constant LDC",
            constantLdcPtx.Contains("mov.u32 %r0, %rpp0;") &&
            !constantLdcPtx.Contains("LDC"));
        var signedByteLdcSection = new rtxdisam.SassSection();
        signedByteLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.S8",
            Operands = { "R1", "c[0x0][0x140]" },
        });
        signedByteLdcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDC.S16",
            Operands = { "R2", "c[0x0][0x142]" },
        });
        signedByteLdcSection.Nodes.Add(
            new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string signedByteLdcPtx = rtxdisam.SassToPtx.Translate(
            signedByteLdcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u32", 0x140, 4) },
            "sm_86");
        Check("SASS to PTX signed narrow LDC sign-extends",
            signedByteLdcPtx.Contains("shr.s32 %r1, %r1, 24;") &&
            signedByteLdcPtx.Contains("shr.s32 %r2, %r2, 16;") &&
            !signedByteLdcPtx.Contains("and.b32 %r1, %r1, 0xff;"));
        var fadd32iSection = new rtxdisam.SassSection();
        fadd32iSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD32I",
            Operands = { "R0", "R1", "1.5" },
        });
        string fadd32iPtx = rtxdisam.SassToPtx.Translate(
            fadd32iSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX FADD32I",
            fadd32iPtx.Contains("mov.b32 %ft0, %r1;") &&
            fadd32iPtx.Contains("mov.b32 %ft1, 0x3fc00000;") &&
            fadd32iPtx.Contains("add.f32 %ft2, %ft0, %ft1;") &&
            fadd32iPtx.Contains("mov.b32 %r0, %ft2;"));
        var fadd32iCcSection = new rtxdisam.SassSection();
        fadd32iCcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD32I",
            Operands = { "R122.CC", "-|R93|", "-117478680" },
            LegacyFloatModifiers = "NEG.ABS",
        });
        fadd32iCcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD",
            Operands = { "R4.CC", "R6", "c[0x0][0x140]" },
        });
        fadd32iCcSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R5", "R0", "c[0x0][0x144]" },
        });
        fadd32iCcSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string fadd32iCcPtx = rtxdisam.SassToPtx.Translate(
            fadd32iCcSection, "k",
            new[] { new rtxdisam.PtxParameter("p0", "u64", 0x140, 8) },
            "sm_50");
        Check("SASS to PTX FADD32I dead CC",
            fadd32iCcPtx.Contains("mov.b32 %ft0, %r93;") &&
            fadd32iCcPtx.Contains("abs.f32 %ft1, %ft0;") &&
            fadd32iCcPtx.Contains("neg.f32 %ft2, %ft1;") &&
            fadd32iCcPtx.Contains("mov.b32 %ft3, 0xf8ff6ae8;") &&
            fadd32iCcPtx.Contains("add.f32 %ft4, %ft2, %ft3;") &&
            fadd32iCcPtx.Contains("mov.b32 %r122, %ft4;"));
        var fadd32iLiveSection = new rtxdisam.SassSection();
        fadd32iLiveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "FADD32I",
            Operands = { "R0.CC", "R1", "1.5" },
        });
        fadd32iLiveSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "IADD.X",
            Operands = { "R2", "R3", "R4" },
        });
        fadd32iLiveSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        bool rejectedLiveCc = false;
        try
        {
            rtxdisam.SassToPtx.Translate(
                fadd32iLiveSection, "k",
                Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        }
        catch (NotSupportedException)
        {
            rejectedLiveCc = true;
        }
        Check("SASS to PTX rejects live FADD32I CC", rejectedLiveCc);
        var ldslkSection = new rtxdisam.SassSection();
        ldslkSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDSLK",
            Operands = { "P1", "R6", "[R0]" },
        });
        ldslkSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STSCUL",
            Guard = "@P1",
            Operands = { "P0", "[R0]", "R3" },
        });
        string ldslkPtx = rtxdisam.SassToPtx.Translate(
            ldslkSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX LDSLK/STSCUL",
            ldslkPtx.Contains("ld.shared.u32 %r6, [") &&
            ldslkPtx.Contains("setp.eq.u32 %p1, 0, 0;") &&
            ldslkPtx.Contains("atom.shared.cas.b32 %rt") &&
            ldslkPtx.Contains("], %r6, %r3;") &&
            ldslkPtx.Contains("setp.eq.u32 %p0, %rt") &&
            ldslkPtx.Contains(", %r6;"));
        var nopSyncGuardSection = new rtxdisam.SassSection();
        nopSyncGuardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "SSY",
            Operands = { "`(.L_1)`" },
        });
        nopSyncGuardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP.S",
            Guard = "@P0",
            BranchTargets = { ".L_1" },
        });
        nopSyncGuardSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "R3" },
        });
        nopSyncGuardSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_1" });
        nopSyncGuardSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string nopSyncPtx = rtxdisam.SassToPtx.Translate(
            nopSyncGuardSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX guarded NOP.S",
            Regex.IsMatch(nopSyncPtx, @"@%p0 bra L_\d+;"));
        var nopSyncNextSection = new rtxdisam.SassSection();
        nopSyncNextSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "NOP.S",
            BranchTargets = { ".L_2" },
        });
        nopSyncNextSection.Nodes.Add(new rtxdisam.SassLabel { Name = ".L_2" });
        nopSyncNextSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        string nopSyncNextPtx = rtxdisam.SassToPtx.Translate(
            nopSyncNextSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX NOP.S to next label",
            !nopSyncNextPtx.Contains("bra L__L_2;"));
        var popcMaskSection = new rtxdisam.SassSection();
        popcMaskSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "POPC",
            Operands = { "R2", "R0", "R0" },
        });
        string popcMaskPtx = rtxdisam.SassToPtx.Translate(
            popcMaskSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX POPC same mask",
            popcMaskPtx.Contains("popc.b32 %r2, %r0;"));
        var popcAndSection = new rtxdisam.SassSection();
        popcAndSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "POPC",
            Operands = { "R2", "R0", "R1" },
        });
        string popcAndPtx = rtxdisam.SassToPtx.Translate(
            popcAndSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX POPC mask",
            popcAndPtx.Contains("and.b32 %rt0, %r0, %r1;") &&
            popcAndPtx.Contains("popc.b32 %r2, %rt0;"));
        var guardedIsetSection = new rtxdisam.SassSection();
        guardedIsetSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISET.GE.U32.AND",
            Guard = "@P1",
            Operands = { "R2", "R6", "0x65", "PT" },
        });
        string guardedIsetPtx = rtxdisam.SassToPtx.Translate(
            guardedIsetSection, "k", Array.Empty<rtxdisam.PtxParameter>(), "sm_50");
        Check("SASS to PTX guarded ISET",
            guardedIsetPtx.Contains("setp.ge.u32 %pt0, %r6, 0x65;") &&
            !guardedIsetPtx.Contains("@%p1 setp.") &&
            guardedIsetPtx.Contains("@%p1 selp.b32 %r2, 1, 0, %pt0;"));
        var isetpSignedSection = new rtxdisam.SassSection();
        isetpSignedSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "ISETP.EQ.AND",
            Operands = { "P0", "PT", "R0", "RZ", "PT" },
        });
        byte[] isetpBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", isetpSignedSection);
        Check("sm35 ISETP signed encoding",
            BinaryPrimitives.ReadUInt64LittleEndian(
                isetpBytes.AsSpan(8)) == 0xdb281c007f9c001eUL,
            BinaryPrimitives.ReadUInt64LittleEndian(
                isetpBytes.AsSpan(8)).ToString("x"));
        var sharedImmediateSection = new rtxdisam.SassSection();
        sharedImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "STS",
            Operands = { "[0x10]", "R6" },
        });
        sharedImmediateSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R6", "[0x8]" },
        });
        var sharedImmediateDecoded = PreVoltaDecoder
            .DecodeSection(rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_30", sharedImmediateSection))
            .ToList();
        Equal("sm30 STS immediate address", "STS [0x10], R6",
            sharedImmediateDecoded[0].Instruction.Text);
        Equal("sm30 LDS immediate address", "LDS R6, [0x8]",
            sharedImmediateDecoded[1].Instruction.Text);
        var dualSection = new rtxdisam.SassSection();
        dualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        dualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R3", "R4" },
        });
        dualSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        ulong dualControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection("sm_35", dualSection));
        Check("kepler MOV/MOV dual issue",
            (dualControl & (1UL << 7)) == 0);
        Check("SM35 sparse MOV/MOV/EXIT sections",
            ((dualControl >> 2) & 0xFF) == 0x10 &&
            ((dualControl >> (2 + 8)) & 0xFF) == 0x24);
        var selfDualSection = new rtxdisam.SassSection();
        selfDualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R1" },
        });
        selfDualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R2", "R2" },
        });
        selfDualSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        ulong selfDualControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection("sm_35", selfDualSection));
        Check("kepler self-move single issue",
            (selfDualControl & (1UL << 7)) != 0);
        var paddingDualSection = new rtxdisam.SassSection();
        paddingDualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        paddingDualSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "RZ", "RZ" },
        });
        ulong paddingDualControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", paddingDualSection));
        Check("kepler MOV/padding single issue",
            (paddingDualControl & (1UL << 7)) != 0);
        var sharedBarrierSection = new rtxdisam.SassSection();
        sharedBarrierSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "LDS",
            Operands = { "R1", "[R2]" },
        });
        sharedBarrierSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        ulong sharedBarrierControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", sharedBarrierSection));
        Check("kepler shared barrier flag",
            (sharedBarrierControl & (1UL << (2 + 4))) != 0);
        var ptxasSimpleSection = new rtxdisam.SassSection();
        ptxasSimpleSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        ptxasSimpleSection.Nodes.Add(new rtxdisam.SassInstruction { Mnemonic = "EXIT" });
        ulong ptxasSimpleControl = BinaryPrimitives.ReadUInt64LittleEndian(
            rtxdisam.PreVoltaAssembler.EncodeSection(
                "sm_35", ptxasSimpleSection));
        Check("sm35 scheduling matches ptxas simple control",
            ptxasSimpleControl == 0x08a0a0a0fca094c0UL,
            $"0x{ptxasSimpleControl:x16}");
        ulong keplerTemplate = 0x08a0a0a0fca010c0UL;
        ulong keplerScheduled = rtxdisam.PreVoltaScheduling.SetKeplerStalls(
            keplerTemplate, new[] { 1, 2, 3, 4, 5, 6, 7 });
        Check("kepler control tag",
            (keplerScheduled & rtxdisam.PreVoltaScheduling.KeplerControlTag) != 0);
        ulong gk104Base = 0x2000000000000007UL;
        for (int gkSlot = 0; gkSlot < 7; gkSlot++)
            gk104Base |= 0x20UL << (4 + gkSlot * 8);
        ulong gk104Scheduled = rtxdisam.PreVoltaScheduling.SetGk104Stalls(
            gk104Base, new[] { 1, 5, 3, 4, 5, 6, 7 });
        Check("gk104 slot 0 delay",
            ((gk104Scheduled >> 4) & 0xFF) == 0x21);
        Check("gk104 slot 1 control delay",
            ((gk104Scheduled >> 12) & 0xFF) == 0x25);
        ulong gk104LongDelay = rtxdisam.PreVoltaScheduling.SetGk104Stalls(
            0x2000000000000007UL, new[] { 32, 0, 0, 0, 0, 0, 0 });
        Check("gk104 32-cycle delay encoding",
            ((gk104LongDelay >> 4) & 0xFF) == 0x00);
        for (int keplerSlot = 0; keplerSlot < 7; keplerSlot++)
        {
            Check($"kepler slot {keplerSlot} stall",
                ((keplerScheduled >> (2 + keplerSlot * 8)) & 0xF) ==
                    (ulong)(keplerSlot + 1));
        }
        var keplerSimpleSection = new rtxdisam.SassSection();
        keplerSimpleSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "MOV",
            Operands = { "R1", "R2" },
        });
        keplerSimpleSection.Nodes.Add(new rtxdisam.SassInstruction
        {
            Mnemonic = "EXIT",
        });
        byte[] keplerSimpleBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_35", keplerSimpleSection);
        ulong keplerSimpleControl = BitConverter.ToUInt64(keplerSimpleBytes, 0);
        Check("kepler exit minimum stall",
            ((keplerSimpleControl >> (2 + 8)) & 0xF) >= 5,
            ((keplerSimpleControl >> (2 + 8)) & 0xF).ToString());
        Check("kepler control op is single issue",
            ((keplerSimpleControl >> (2 + 8 + 5)) & 1) == 1);
        byte[] gk104SimpleBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_32", keplerSimpleSection);
        ulong gk104Control = BitConverter.ToUInt64(gk104SimpleBytes, 0);
        Check("sm32 exit stall",
            ((gk104Control >> (2 + 8)) & 0x1F) >= 5,
            ((gk104Control >> (2 + 8)) & 0x1F).ToString());
        byte[] sm30ControlBytes = rtxdisam.PreVoltaAssembler.EncodeSection(
            "sm_30", keplerSimpleSection);
        ulong sm30Control = BitConverter.ToUInt64(sm30ControlBytes, 0);
        Check("sm30 keeps sched opcode",
            (sm30Control & 0xF000000000000007UL) == 0x2000000000000007UL);
        Check("sm30 GK104 exit stall",
            ((sm30Control >> (4 + 8)) & 0x1F) >= 5,
            ((sm30Control >> (4 + 8)) & 0x1F).ToString());
        Check("sm30 ptxas probe control",
            sm30Control == 0x200000000002f307UL,
            $"0x{sm30Control:x16}");
        Check("pre-volta dependency stall",
            rtxdisam.PreVoltaScheduling.DependencyStall("IADD", "IADD") == 6 &&
            rtxdisam.PreVoltaScheduling.DependencyStall(
                "XMAD", "STG.E", maxwell: true) == 1);
        Check("pre-volta dependency stall distance",
            rtxdisam.PreVoltaScheduling.DependencyStall("IADD", "IADD", 6) == 0);
        var stalls = rtxdisam.PreVoltaScheduling.EstimateStalls(
            new[] { "IADD", "IADD", "IADD" });
        Check("pre-volta slot stall estimate",
            stalls[0] == 0 && stalls[1] == 6 && stalls[2] == 6);
        Check("pre-volta memory timing",
            rtxdisam.PreVoltaScheduling.TryGet("LDG.E", out var ldgTiming) &&
            ldgTiming.ReadLatency == 4 &&
            ldgTiming.BarrierLatency == 200);
        Check("pre-volta memory dependency stall",
            rtxdisam.PreVoltaScheduling.DependencyStall("LDG.E", "IADD") == 4);
        ulong barrierControl = rtxdisam.PreVoltaScheduling.SetMaxwellBarriers(
            0UL, 0, writeBarrier: 1, readBarrier: 2, waitMask: 4);
        Check("pre-volta barrier encoding",
            ((barrierControl >> 5) & 7) == 0 &&
            ((barrierControl >> 8) & 7) == 1 &&
            ((barrierControl >> 11) & 0x3F) == 4);
        ulong scheduledControl = rtxdisam.PreVoltaScheduling.SetMaxwellStalls(
            0x001fc800fe2007f6UL, stalls);
        Check("pre-volta control stall encoding",
            (scheduledControl & 0xF) == 0 &&
            ((scheduledControl >> 21) & 0xF) == 6 &&
            ((scheduledControl >> 42) & 0xF) == 6);
        var crossBundleStalls = rtxdisam.PreVoltaScheduling.EstimateStalls(
            new[] { "IADD", "IADD" }, "IADD");
        Check("pre-volta cross-bundle stall",
            crossBundleStalls[0] == 6 && crossBundleStalls[1] == 6);
        Check("pre-volta register dependency stall",
            rtxdisam.PreVoltaScheduling.DependencyStall(
                "IADD", new[] { "R1", "R2", "R3" },
                "IADD", new[] { "R4", "R1", "R3" }) == 6);
        Check("pre-volta independent stall",
            rtxdisam.PreVoltaScheduling.DependencyStall(
                "IADD", new[] { "R1", "R2", "R3" },
                "IADD", new[] { "R4", "R5", "R6" }) == 0);
        Check("pre-volta memory dependency stall",
            rtxdisam.PreVoltaScheduling.DependencyStall(
                "LDG.E", new[] { "R1", "[R2]" },
                "IADD", new[] { "R3", "R1", "R4" }) == 4);
        PreVoltaDecoder.UseArchitecture("sm_50");
        Equal("sm50 mov constant",
            "MOV R1, c[0x0][0x20]",
            PreVoltaDecoder.Decode(0x4c98078000870001).Text);
        Equal("sm50 mov constant 2",
            "MOV R2, c[0x0][0x140]",
            PreVoltaDecoder.Decode(0x4c98078005070002).Text);
        Equal("sm50 mov constant 3",
            "MOV R3, c[0x0][0x144]",
            PreVoltaDecoder.Decode(0x4c98078005170003).Text);
        Equal("sm50 xmad",
            "XMAD R5, R0, 0x5, R5",
            PreVoltaDecoder.Decode(0x3600028000570005).Text);
        Equal("sm50 xmad psl",
            "XMAD.PSL R0, R0.H1, 0x5, R5",
            PreVoltaDecoder.Decode(0x3620029000570000).Text);
        Equal("sm50 xmad reuse",
            "XMAD R5, R0.reuse, 0x5, R5",
            PreVoltaDecoder.Decode(0x3600028000570005, 0, 1).Text);
        Equal("sm50 xmad rc reuse",
            "XMAD R5, R0, 0x5, R5.reuse",
            PreVoltaDecoder.Decode(0x3600028000570005, 0, 4).Text);
        Equal("sm50 ldg u8",
            "LDG.E.U8 R7, [R2+0x12]",
            PreVoltaDecoder.Decode(0xeed0200001270207).Text);
        Equal("sm50 ldg 64",
            "LDG.E.64 R4, [R2+0x8]",
            PreVoltaDecoder.Decode(0xeed5200000870204).Text);
        Equal("sm50 fadd immediate",
            "FADD R10, R9, 1.5",
            PreVoltaDecoder.Decode(0x3858003fc007090a).Text);
        Equal("sm50 i2f",
            "I2F.F32.S32 R9, -R8",
            PreVoltaDecoder.Decode(0x5cb8200000872a09).Text);
        Equal("sm50 f2i",
            "F2I.S32.F32.TRUNC R11, R11",
            PreVoltaDecoder.Decode(0x5cb0018000b71a0b).Text);
        Equal("sm50 iadd negate",
            "IADD R6, -R6, R8",
            PreVoltaDecoder.Decode(0x5c12000000870606).Text);
        Equal("sm50 shr u32",
            "SHR.U32 R6, R6, 0x2",
            PreVoltaDecoder.Decode(0x3828000000270606).Text);
        Equal("sm50 flo u32",
            "FLO.U32 R10, R9",
            PreVoltaDecoder.Decode(0x5c3000000097000a).Text);
        Equal("sm50 branch target",
            "BRA 0x78",
            PreVoltaDecoder.Decode(0xe2400fffff87000f, 0x78).Text);
        Equal("sm50 ldg",
            "LDG.E R0, [R2]",
            PreVoltaDecoder.Decode(0xeed4200000070200).Text);
        Equal("sm50 stg",
            "STG.E [R2], R0",
            PreVoltaDecoder.Decode(0xeedc200000070200).Text);
        Equal("sm50 exit",
            "EXIT",
            PreVoltaDecoder.Decode(0xe30000000007000f).Text);

        byte[] sm50Bundle = new byte[32];
        BinaryPrimitives.WriteUInt64LittleEndian(sm50Bundle.AsSpan(8), 0x4c98078000870001);
        BinaryPrimitives.WriteUInt64LittleEndian(sm50Bundle.AsSpan(16), 0x4c98078005070002);
        BinaryPrimitives.WriteUInt64LittleEndian(sm50Bundle.AsSpan(24), 0x4c98078005170003);
        var sm50Decoded = PreVoltaDecoder.DecodeSection(sm50Bundle).ToList();
        Check("sm50 bundle instruction count", sm50Decoded.Count == 3);
        Check("sm50 bundle addresses",
            sm50Decoded[0].Address == 8 &&
            sm50Decoded[1].Address == 16 &&
            sm50Decoded[2].Address == 24);
        Equal("sm50 bundle first instruction",
            "MOV R1, c[0x0][0x20]",
            sm50Decoded[0].Instruction.Text);

        InstructionDecoder.UseArchitecture(75, "13.4");
        Check("sm75 table architecture",
            InstructionDecoder.Table.Architecture == "sm_75",
            InstructionDecoder.Table.Architecture);
        Equal("sm75 hfma2 immediate order",
            "HFMA2 R0, R7, 2.5, 1.5, R0",
            Decode("0fe200000000000041003e0007007831").Text);
        Equal("sm75 frnd f32 default type",
            "FRND.TRUNC R2, R2",
            Decode("000e22000020d0000000000200027307").Text);
        Equal("sm75 13-bit opcode uniform add",
            "IADD3 R3, R2, UR4, R3",
            Decode("000fc6000fffe0030000000402037c10").Text);
        InstructionDecoder.UseArchitecture(86);

        InstructionDecoder.UseArchitecture(80);
        Check("sm80 table architecture",
            InstructionDecoder.Table.Architecture == "sm_80",
            InstructionDecoder.Table.Architecture);
        Equal("sm80 f2fp pack_ab",
            "F2FP.PACK_AB R5, R5, R2",
            Decode("0004fca00000000ff000000020505723e").Text);
        InstructionDecoder.UseArchitecture(86);

        InstructionDecoder.UseArchitecture(89);
        Equal("sm89 f2fp keeps explicit f32 type",
            "F2FP.F16.F32.PACK_AB R11, RZ, R11",
            Decode("000fc000000000ff0000000bff0b723e").Text);
        Equal("sm89 negative absolute ret",
            "RET.ABS.NODEC R20 -0x170",
            Decode("000fc00003e3fffffffffe9014007950").Text);
        InstructionDecoder.UseArchitecture(86);

        InstructionDecoder.UseArchitecture(90);
        Check("sm90 table architecture",
            InstructionDecoder.Table.Architecture == "sm_90",
            InstructionDecoder.Table.Architecture);
        Equal("sm90 descriptor ldg",
            "LDG.E.128 R12, desc[UR4][R2.64]",
            Decode("000ea2000c1e1d0000000004020c7981").Text);
        Equal("sm90 imad high pseudo modifier",
            "IMAD.HI.U32 R12, R10, R15, RZ",
            Decode("000fc000078e00ff0000000f0a0c7227").Text);
        InstructionDecoder.UseArchitecture(86);

        InstructionDecoder.UseArchitecture(107);
        Equal("sm107 sel 64 modifier",
            "SEL.64 R10, R10, R6, !P0",
            Decode("000fc00004000000000000060a0a7607").Text);
        InstructionDecoder.UseArchitecture(86);

        foreach (string architecture in new[] { "sm_87", "sm_88", "sm_89", "sm_90a" })
        {
            InstructionDecoder.UseArchitecture(architecture);
            Check($"{architecture} table architecture",
                InstructionDecoder.Table.Architecture == architecture,
                InstructionDecoder.Table.Architecture);
        }
        foreach (string architecture in new[]
        {
            "sm_100", "sm_101", "sm_103", "sm_107", "sm_110", "sm_120", "sm_121",
        })
        {
            InstructionDecoder.UseArchitecture(architecture);
            string expected = architecture;
            Check($"{architecture} table architecture",
                InstructionDecoder.Table.Architecture == expected,
                InstructionDecoder.Table.Architecture);
        }
        InstructionDecoder.UseArchitecture(86);
    }
}
