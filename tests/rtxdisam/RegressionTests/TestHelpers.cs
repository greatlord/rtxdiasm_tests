using System.Globalization;
using nvDismDecode;
using rtxdisam;

namespace RegressionTests;

internal static class TestHelpers
{
    public static (ulong Low, ulong High) Parse128(string hex)
    {
        UInt128 value = UInt128.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
        return ((ulong)value, (ulong)(value >> 64));
    }

    public static DecodedInstruction Decode(string hex, ulong address = 0)
    {
        var (low, high) = Parse128(hex);
        return InstructionDecoder.Decode(low, high, address);
    }

    public static CubinSection SpillSection()
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

    public static List<(int Offset, ulong Address, DecodedInstruction Instruction)> SpillItems(
        params (string Text, ulong Address)[] texts) =>
        texts.Select(item => (
            (int)item.Address,
            item.Address,
            new DecodedInstruction { Text = item.Text, Address = item.Address }))
            .ToList();

    public static CubinSection RelocationSection(
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

    public static string? ApplyRelocation(
        CubinSection section, string text, bool nvJsonStyle = false) =>
        rtxdisam.Program.TryApplyRelocation(
            section, 0, text, new Dictionary<(CubinSection, ulong), string>(),
            numericTargets: nvJsonStyle, nvJsonStyle: nvJsonStyle);
}
