using System.Buffers.Binary;
using System.Text;
using K4os.Compression.LZ4;

namespace RegressionTests;

internal static class Fixtures
{
    internal static byte[] BuildOldFatbin(
        ushort kind, byte[] payload, ulong binInfo, ulong uncompressedSize,
        uint arch, string? identifier = null, string? ptxasOptions = null)
    {
        byte[] identifierBytes = Encoding.UTF8.GetBytes(identifier ?? "");
        byte[] optionBytes = Encoding.UTF8.GetBytes(ptxasOptions ?? "");
        int cursor = 64;
        int descriptorOffset = cursor;
        cursor += 8; // ptxas descriptor
        int optionOffset = cursor;
        if (ptxasOptions != null)
            cursor = Align(cursor + optionBytes.Length + 1, 8);
        int identifierOffset = 0;
        if (identifier != null)
        {
            identifierOffset = cursor;
            cursor = Align(cursor + identifierBytes.Length + 1, 8);
        }
        int headerSize = cursor;
        int padded = (payload.Length + 7) & ~7;
        long total = headerSize + padded;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(0xBA55ED50u);
        writer.Write((ushort)1);
        writer.Write((ushort)16);
        writer.Write((ulong)total);

        writer.Write(kind);
        writer.Write((ushort)0x0101);
        writer.Write((uint)headerSize);
        writer.Write((long)padded);
        writer.Write((uint)payload.Length);
        writer.Write((uint)descriptorOffset);
        writer.Write((ushort)7); // CUDA 10.2 code version [1, 7]
        writer.Write((ushort)1);
        writer.Write(arch);
        writer.Write((uint)identifierOffset);
        writer.Write((uint)identifierBytes.Length);
        writer.Write(binInfo);
        writer.Write(0UL);
        writer.Write(uncompressedSize);

        byte[] extra = new byte[headerSize - 64];
        BinaryPrimitives.WriteUInt32LittleEndian(
            extra.AsSpan(descriptorOffset - 64),
            ptxasOptions != null ? (uint)optionOffset : 0);
        BinaryPrimitives.WriteUInt32LittleEndian(
            extra.AsSpan(descriptorOffset - 64 + 4), (uint)optionBytes.Length);
        if (ptxasOptions != null)
            optionBytes.CopyTo(extra, optionOffset - 64);
        if (identifier != null)
            identifierBytes.CopyTo(extra, identifierOffset - 64);
        writer.Write(extra);
        writer.Write(payload);
        for (int i = payload.Length; i < padded; i++)
            writer.Write((byte)0);
        writer.Flush();
        return stream.ToArray();
    }

    internal static byte[] BuildFatbin(
        ushort kind, byte[] payload, ulong binInfo, ulong uncompressedSize,
        uint arch = 86, int? declaredPayloadSize = null, string? identifier = null,
        string? ptxasOptions = null)
    {
        byte[] identifierBytes = Encoding.UTF8.GetBytes(identifier ?? "");
        byte[] optionBytes = Encoding.UTF8.GetBytes(ptxasOptions ?? "");

        int cursor = 64;
        int identifierOffset = 0;
        if (identifier != null)
        {
            identifierOffset = cursor;
            cursor = Align(cursor + identifierBytes.Length + 1, 8);
        }

        int descriptorOffset = 0;
        int optionOffset = 0;
        if (ptxasOptions != null)
        {
            descriptorOffset = cursor;
            cursor += 8;
            optionOffset = cursor;
            cursor = Align(cursor + optionBytes.Length + 1, 8);
        }
        int headerSize = cursor;

        int padded = (payload.Length + 7) & ~7;
        long total = headerSize + padded;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(0xBA55ED50u);
        writer.Write((ushort)1);
        writer.Write((ushort)16);
        writer.Write((ulong)total);

        writer.Write(kind);
        writer.Write((ushort)0x0101);
        writer.Write((uint)headerSize);
        writer.Write((long)padded);
        writer.Write(declaredPayloadSize ?? payload.Length);
        writer.Write((uint)descriptorOffset);
        writer.Write((ushort)0);
        writer.Write((ushort)8);
        writer.Write(arch);
        writer.Write((uint)identifierOffset);
        writer.Write((uint)identifierBytes.Length);
        writer.Write(binInfo);
        writer.Write(0UL);
        writer.Write(uncompressedSize);

        if (headerSize > 64)
        {
            byte[] extra = new byte[headerSize - 64];
            if (identifierOffset != 0)
                identifierBytes.CopyTo(extra, identifierOffset - 64);
            if (descriptorOffset != 0)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(
                    extra.AsSpan(descriptorOffset - 64), (uint)optionOffset);
                BinaryPrimitives.WriteUInt32LittleEndian(
                    extra.AsSpan(descriptorOffset - 64 + 4), (uint)optionBytes.Length);
                optionBytes.CopyTo(extra, optionOffset - 64);
            }
            writer.Write(extra);
        }

        writer.Write(payload);
        for (int i = payload.Length; i < padded; i++)
            writer.Write((byte)0);
        writer.Flush();
        return stream.ToArray();
    }

    internal static byte[] BuildPe(byte[] fatbin)
    {
        const int peOffset = 0x40;
        const int optionalSize = 0xF0;
        int sectionTable = peOffset + 24 + optionalSize;
        int rawOffset = Align(sectionTable + 40, 0x200);
        var data = new byte[rawOffset + fatbin.Length];

        data[0] = (byte)'M';
        data[1] = (byte)'Z';
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x3C), peOffset);
        data[peOffset] = (byte)'P';
        data[peOffset + 1] = (byte)'E';
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 16), optionalSize);

        Encoding.ASCII.GetBytes(".nv_fatb").CopyTo(data, sectionTable);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 16), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 20), (uint)rawOffset);

        fatbin.CopyTo(data, rawOffset);
        return data;
    }

    internal static byte[] BuildCoff(byte[] fatbin)
    {
        int sectionTable = 20;
        int rawOffset = 0x200;
        var data = new byte[rawOffset + fatbin.Length];

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(0), 0x8664); // x64
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(2), 1);      // sections
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 0);    // optional header

        Encoding.ASCII.GetBytes(".nv_fatb").CopyTo(data, sectionTable);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 8), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 16), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 20), (uint)rawOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(sectionTable + 36), 0x40000040);

        fatbin.CopyTo(data, rawOffset);
        return data;
    }

    internal static byte[] BuildPeWithWrapper(byte[] fatbin)
    {
        const int peOffset = 0x40;
        const int optionalSize = 0xF0;
        const ulong imageBase = 0x140000000;
        int sectionTable = peOffset + 24 + optionalSize;
        int textOffset = Align(sectionTable + 2 * 40, 0x200);
        int wrapperOffset = Align(textOffset + fatbin.Length, 0x200);
        var data = new byte[wrapperOffset + 24];

        data[0] = (byte)'M';
        data[1] = (byte)'Z';
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x3C), peOffset);
        data[peOffset] = (byte)'P';
        data[peOffset + 1] = (byte)'E';
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 2), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 16), optionalSize);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 24), 0x20B);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(peOffset + 24 + 0x18), imageBase);

        WritePeSection(data, sectionTable, ".text",
            fatbin.Length, 0x1000, fatbin.Length, textOffset, 0x60000020);
        WritePeSection(data, sectionTable + 40, ".nvFatBi",
            24, 0x2000, 24, wrapperOffset, 0x40000040);

        fatbin.CopyTo(data, textOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(wrapperOffset), 0x466243B1);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(wrapperOffset + 4), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(wrapperOffset + 8), imageBase + 0x1000);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 16), 0);
        return data;
    }

    internal static void WritePeSection(
        byte[] data, int offset, string name, int virtualSize, int virtualAddress,
        int rawSize, int rawPointer, int characteristics)
    {
        Encoding.ASCII.GetBytes(name).CopyTo(data, offset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 8), (uint)virtualSize);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 12), (uint)virtualAddress);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 16), (uint)rawSize);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 20), (uint)rawPointer);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 36), (uint)characteristics);
    }

    internal static byte[] BuildElfWithCuinfo(uint arch, uint eFlags = 0)
    {
        byte[] shstr = Encoding.ASCII.GetBytes(
            "\0.shstrtab\0.note.nv.cuinfo\0");
        const int headerSize = 64;
        int shstrOffset = headerSize;
        int noteOffset = Align(shstrOffset + shstr.Length, 4);

        byte[] note = new byte[12 + 12 + 8];
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(0), 12);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(4), 8);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(8), 0x3e8);
        Encoding.ASCII.GetBytes("NVIDIA Corp\0").CopyTo(note, 12);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(24), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(26), (ushort)arch);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(28), 0x86);

        int sectionOffset = Align(noteOffset + note.Length, 8);
        var data = new byte[sectionOffset + 3 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2;
        data[5] = 1;
        data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);   // ET_REL
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 190); // CUDA
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x30), eFlags);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        note.CopyTo(data, noteOffset);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += 64;
        WriteSection(data, sh, 11, 7, 0, noteOffset, note.Length, 4, 0);
        return data;
    }

    internal static byte[] BuildElfWithTkinfo()
    {
        byte[] shstr = Encoding.ASCII.GetBytes(
            "\0.shstrtab\0.note.nv.tkinfo\0");
        const int headerSize = 64;
        int shstrOffset = headerSize;
        int noteOffset = Align(shstrOffset + shstr.Length, 4);

        byte[] first = BuildTkinfoNote("nvlink");
        byte[] second = BuildTkinfoNote("ptxas");
        int firstSize = Align(first.Length, 4);
        byte[] notes = new byte[firstSize + second.Length];
        first.CopyTo(notes, 0);
        second.CopyTo(notes, firstSize);

        int sectionOffset = Align(noteOffset + notes.Length, 8);
        var data = new byte[sectionOffset + 3 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2;
        data[5] = 1;
        data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);   // ET_REL
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 190); // CUDA
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        notes.CopyTo(data, noteOffset);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += 64;
        WriteSection(data, sh, 11, 7, 0, noteOffset, notes.Length, 4, 0);
        return data;
    }

    internal static byte[] BuildTkinfoNote(string tool)
    {
        string name = tool;
        string version = "v";
        string branch = "b";
        string options = "o";
        int nameOffset = 1;
        int versionOffset = nameOffset + name.Length + 1;
        int branchOffset = versionOffset + version.Length + 1;
        int optionsOffset = branchOffset + branch.Length + 1;

        var header = new byte[24];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0), 2);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), (uint)nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            header.AsSpan(12), (uint)versionOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            header.AsSpan(16), (uint)branchOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            header.AsSpan(20), (uint)optionsOffset);
        var payload = new List<byte>(header);
        payload.AddRange(Encoding.UTF8.GetBytes(
            "\0" + name + "\0" + version + "\0" + branch + "\0" +
            options + "\0"));

        var note = new byte[24 + payload.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(0), 12);
        BinaryPrimitives.WriteUInt32LittleEndian(
            note.AsSpan(4), (uint)payload.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(8), 2000);
        Encoding.ASCII.GetBytes("NVIDIA Corp\0").CopyTo(note, 12);
        payload.CopyTo(note, 24);
        return note;
    }

    internal static byte[] BuildBrxElf()
    {
        byte[] shstr = Encoding.ASCII.GetBytes(
            "\0.shstrtab\0.text.foo\0.nv.info.foo\0");
        byte[] text = new byte[0x200];
        byte[] info = new byte[4 + 24 + 8];
        info[0] = 4;
        info[1] = 52;
        BinaryPrimitives.WriteUInt16LittleEndian(info.AsSpan(2), 24);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(4), 0x10);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(8), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(12), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(16), 0x100);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(20), 0x120);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(24), 0x140);
        info[28] = 4;
        info[29] = 102;
        BinaryPrimitives.WriteUInt16LittleEndian(info.AsSpan(30), 4);

        const int headerSize = 64;
        int shstrOffset = headerSize;
        int textOffset = Align(shstrOffset + shstr.Length, 16);
        int infoOffset = Align(textOffset + text.Length, 8);
        int sectionOffset = Align(infoOffset + info.Length, 8);

        var data = new byte[sectionOffset + 4 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2;
        data[5] = 1;
        data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 62);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        text.CopyTo(data, textOffset);
        info.CopyTo(data, infoOffset);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0);
        sh += 64;
        WriteSection(data, sh, 11, 1, 0x6, textOffset, text.Length, 16, 0);
        sh += 64;
        WriteSection(data, sh, 21, 1, 0, infoOffset, info.Length, 4, 0);
        return data;
    }

    internal static byte[] BuildRelocElf()
    {
        byte[] shstr = Encoding.ASCII.GetBytes(
            "\0.shstrtab\0.text\0.rela.text\0.symtab\0.strtab\0");
        byte[] strtab = Encoding.ASCII.GetBytes("\0f\0.text\0g\0");
        byte[] text = new byte[48];
        byte[] symtab = new byte[5 * 24];
        byte[] rela = new byte[3 * 24];

        void Symbol(int index, uint name, byte info, ushort section,
            ulong value, ulong size)
        {
            int offset = index * 24;
            BinaryPrimitives.WriteUInt32LittleEndian(
                symtab.AsSpan(offset), name);
            symtab[offset + 4] = info;
            BinaryPrimitives.WriteUInt16LittleEndian(
                symtab.AsSpan(offset + 6), section);
            BinaryPrimitives.WriteUInt64LittleEndian(
                symtab.AsSpan(offset + 8), value);
            BinaryPrimitives.WriteUInt64LittleEndian(
                symtab.AsSpan(offset + 16), size);
        }

        Symbol(1, 1, 0x12, 2, 0, 16);
        Symbol(2, 3, 0x03, 2, 0, 0);
        Symbol(3, 1, 0x12, 2, 16, 16);
        Symbol(4, 9, 0x12, 2, 32, 16);

        void Relocation(int index, ulong offset, uint symbol, uint type,
            long addend)
        {
            int entry = index * 24;
            BinaryPrimitives.WriteUInt64LittleEndian(
                rela.AsSpan(entry), offset);
            BinaryPrimitives.WriteUInt64LittleEndian(
                rela.AsSpan(entry + 8), ((ulong)symbol << 32) | type);
            BinaryPrimitives.WriteInt64LittleEndian(
                rela.AsSpan(entry + 16), addend);
        }

        Relocation(0, 0, 1, 56, 0x20);
        Relocation(1, 16, 3, 57, 0x20);
        Relocation(2, 32, 4, 58, 0);

        const int headerSize = 64;
        int shstrOffset = headerSize;
        int strtabOffset = Align(shstrOffset + shstr.Length, 8);
        int symtabOffset = Align(strtabOffset + strtab.Length, 8);
        int textOffset = Align(symtabOffset + symtab.Length, 16);
        int relaOffset = Align(textOffset + text.Length, 8);
        int sectionOffset = Align(relaOffset + rela.Length, 8);

        var data = new byte[sectionOffset + 6 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2;
        data[5] = 1;
        data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);  // ET_REL
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 62);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 6);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        strtab.CopyTo(data, strtabOffset);
        symtab.CopyTo(data, symtabOffset);
        text.CopyTo(data, textOffset);
        rela.CopyTo(data, relaOffset);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0);
        sh += 64;
        WriteSection(data, sh, 11, 1, 0x6, textOffset, text.Length, 16, 0);
        sh += 64;
        WriteSection(data, sh, 17, 4, 0, relaOffset, rela.Length, 8, 4, 2, 24);
        sh += 64;
        WriteSection(data, sh, 29, 2, 0, symtabOffset, symtab.Length, 8, 5, 1, 24);
        sh += 64;
        WriteSection(data, sh, 38, 3, 0, strtabOffset, strtab.Length, 1, 0);
        return data;
    }

    internal static byte[] BuildElfWithFatbin(byte[] fatbin, string sectionName = ".nv_fatbin")
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0" + sectionName + "\0");
        const int headerSize = 64;
        int shstrOffset = headerSize;
        int fatbinOffset = Align(shstrOffset + shstr.Length, 8);
        int sectionOffset = Align(fatbinOffset + fatbin.Length, 8);

        var data = new byte[sectionOffset + 3 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2; // ELFCLASS64
        data[5] = 1; // little endian
        data[6] = 1; // version

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);   // ET_REL
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 62);  // x86-64
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        fatbin.CopyTo(data, fatbinOffset);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += 64;
        WriteSection(data, sh, 11, 1, 0, fatbinOffset, fatbin.Length, 8, 0);
        return data;
    }

    internal static byte[] BuildElfWithWrapper(byte[] fatbin)
    {
        byte[] shstr = Encoding.ASCII.GetBytes(
            "\0.shstrtab\0.rodata\0.nvFatBinSegment\0");
        const int headerSize = 64;
        int shstrOffset = headerSize;
        int fatbinOffset = Align(shstrOffset + shstr.Length, 8);
        int wrapperOffset = Align(fatbinOffset + fatbin.Length, 8);
        int sectionOffset = Align(wrapperOffset + 24, 8);

        var data = new byte[sectionOffset + 4 * 64];
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2;
        data[5] = 1;
        data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 62);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(40), (ulong)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(60), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(62), 1);

        shstr.CopyTo(data, shstrOffset);
        fatbin.CopyTo(data, fatbinOffset);

        // __fatBinC_Wrapper_t { magic, version, data VA, filename }
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(wrapperOffset), 0x466243B1);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(wrapperOffset + 4), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 8), 0x2000);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 16), 0);

        int sh = sectionOffset;
        WriteSection(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        WriteSection(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += 64;
        WriteSection(data, sh, 11, 1, 0x6, fatbinOffset, fatbin.Length, 8, 0); sh += 64;
        WriteSection(data, sh, 19, 1, 0, wrapperOffset, 24, 8, 0);

        // Patch the two section virtual addresses.
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(sectionOffset + 2 * 64 + 16), 0x2000);
        BinaryPrimitives.WriteUInt64LittleEndian(
            data.AsSpan(sectionOffset + 3 * 64 + 16), 0x3000);
        return data;
    }

    internal static void WriteSection(
        byte[] data, int offset, uint name, uint type, ulong flags, int fileOffset,
        int size, int align, int link, int info = 0, int entrySize = 0)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), name);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 4), type);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 8), flags);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 24), (ulong)fileOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 32), (ulong)size);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 40), (uint)link);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 44), (uint)info);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 48), (ulong)align);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 56), (ulong)entrySize);
    }

    internal static byte[] BuildPe32(byte[] fatbin)
    {
        const int peOffset = 0x40;
        const int optionalSize = 0xE0; // PE32
        int sectionTable = peOffset + 24 + optionalSize;
        int rawOffset = Align(sectionTable + 40, 0x200);
        var data = new byte[rawOffset + fatbin.Length];

        data[0] = (byte)'M';
        data[1] = (byte)'Z';
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x3C), peOffset);
        data[peOffset] = (byte)'P';
        data[peOffset + 1] = (byte)'E';
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4), 0x014C); // i386
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 16), optionalSize);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 24), 0x10B); // PE32
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(peOffset + 24 + 0x1C), 0x00400000);

        Encoding.ASCII.GetBytes(".nv_fatb").CopyTo(data, sectionTable);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 16), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 20), (uint)rawOffset);

        fatbin.CopyTo(data, rawOffset);
        return data;
    }

    internal static byte[] BuildCoff32(byte[] fatbin)
    {
        int sectionTable = 20;
        int rawOffset = 0x200;
        var data = new byte[rawOffset + fatbin.Length];

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(0), 0x014C); // i386
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 0);

        Encoding.ASCII.GetBytes(".nv_fatb").CopyTo(data, sectionTable);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 8), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 16), (uint)fatbin.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 20), (uint)rawOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionTable + 36), 0x40000040);

        fatbin.CopyTo(data, rawOffset);
        return data;
    }

    internal static byte[] BuildElf32WithFatbin(byte[] fatbin, string sectionName = ".nv_fatbin")
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0" + sectionName + "\0");
        const int headerSize = 52;
        const int shEntSize = 40;
        int shstrOffset = headerSize;
        int fatbinOffset = Align(shstrOffset + shstr.Length, 4);
        int sectionOffset = Align(fatbinOffset + fatbin.Length, 4);

        var data = new byte[sectionOffset + 3 * shEntSize];
        data[0] = 0x7F; data[1] = (byte)'E'; data[2] = (byte)'L'; data[3] = (byte)'F';
        data[4] = 1; // ELFCLASS32
        data[5] = 1; data[6] = 1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);   // ET_REL
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 3);   // EM_386
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEntSize);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);

        shstr.CopyTo(data, shstrOffset);
        fatbin.CopyTo(data, fatbinOffset);

        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEntSize;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEntSize;
        WriteSection32(data, sh, 11, 1, 0, fatbinOffset, fatbin.Length, 4, 0);
        return data;
    }

    internal static void WriteSection32(
        byte[] data, int offset, uint name, uint type, uint flags, int fileOffset,
        int size, uint align, uint link, uint info = 0, uint entrySize = 0)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), name);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 4), type);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 8), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 12), 0); // addr
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 16), (uint)fileOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 20), (uint)size);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 24), link);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 28), info);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 32), align);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 36), entrySize);
    }

    internal static byte[] BuildPe32WithWrapper(byte[] fatbin)
    {
        const int peOffset = 0x40;
        const int optionalSize = 0xE0; // PE32
        const uint imageBase = 0x00400000;
        int sectionTable = peOffset + 24 + optionalSize;
        int textOffset = Align(sectionTable + 2 * 40, 0x200);
        int wrapperOffset = Align(textOffset + fatbin.Length, 0x200);
        var data = new byte[wrapperOffset + 24];

        data[0] = (byte)'M'; data[1] = (byte)'Z';
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(0x3C), peOffset);
        data[peOffset] = (byte)'P'; data[peOffset + 1] = (byte)'E';
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4), 0x014C);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 2), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 4 + 16), optionalSize);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(peOffset + 24), 0x10B);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(peOffset + 24 + 0x1C), imageBase);

        WritePeSection(data, sectionTable, ".text", fatbin.Length, 0x1000, fatbin.Length, textOffset, 0x60000020);
        WritePeSection(data, sectionTable + 40, ".nvFatBi", 24, 0x2000, 24, wrapperOffset, 0x40000040);

        fatbin.CopyTo(data, textOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(wrapperOffset), 0x466243B1);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(wrapperOffset + 4), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 8), imageBase + 0x1000u);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 16), 0);
        return data;
    }

    internal static byte[] BuildElf32WithCuinfo(uint arch, uint eFlags = 0)
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0.note.nv.cuinfo\0");
        const int headerSize = 52;
        const int shEnt = 40;
        int shstrOffset = headerSize;
        int noteOffset = Align(shstrOffset + shstr.Length, 4);
        byte[] note = new byte[12 + 12 + 8];
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(0), 12);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(4), 8);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(8), 0x3e8);
        Encoding.ASCII.GetBytes("NVIDIA Corp\0").CopyTo(note, 12);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(24), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(26), (ushort)arch);
        BinaryPrimitives.WriteUInt16LittleEndian(note.AsSpan(28), 0x86);

        int sectionOffset = Align(noteOffset + note.Length, 4);
        var data = new byte[sectionOffset + 3 * shEnt];
        data[0]=0x7F; data[1]=(byte)'E'; data[2]=(byte)'L'; data[3]=(byte)'F';
        data[4]=1; data[5]=1; data[6]=1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 190);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(0x24), eFlags);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEnt);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);
        shstr.CopyTo(data, shstrOffset);
        note.CopyTo(data, noteOffset);
        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEnt;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEnt;
        WriteSection32(data, sh, 11, 7, 0, noteOffset, note.Length, 4, 0);
        return data;
    }

    internal static byte[] BuildElf32WithWrapper(byte[] fatbin)
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0.rodata\0.nvFatBinSegment\0");
        const int headerSize = 52;
        const int shEnt = 40;
        int shstrOffset = headerSize;
        int fatbinOffset = Align(shstrOffset + shstr.Length, 4);
        int wrapperOffset = Align(fatbinOffset + fatbin.Length, 4);
        int sectionOffset = Align(wrapperOffset + 24, 4);

        var data = new byte[sectionOffset + 4 * shEnt];
        data[0]=0x7F; data[1]=(byte)'E'; data[2]=(byte)'L'; data[3]=(byte)'F';
        data[4]=1; data[5]=1; data[6]=1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEnt);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);

        shstr.CopyTo(data, shstrOffset);
        fatbin.CopyTo(data, fatbinOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(wrapperOffset), 0x466243B1);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(wrapperOffset + 4), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 8), 0x2000);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(wrapperOffset + 16), 0);

        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEnt;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEnt;
        WriteSection32(data, sh, 11, 1, 0x6, fatbinOffset, fatbin.Length, 4, 0); sh += shEnt;
        WriteSection32(data, sh, 19, 1, 0, wrapperOffset, 24, 4, 0);
        // Patch section virtual addresses (offset 12 in ELF32 shdr)
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionOffset + 2 * shEnt + 12), 0x2000);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(sectionOffset + 3 * shEnt + 12), 0x3000);
        return data;
    }

    internal static byte[] BuildElf32WithTkinfo()
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0.note.nv.tkinfo\0");
        const int headerSize = 52;
        const int shEnt = 40;
        int shstrOffset = headerSize;
        int noteOffset = Align(shstrOffset + shstr.Length, 4);
        byte[] first = BuildTkinfoNote("nvlink");
        byte[] second = BuildTkinfoNote("ptxas");
        int firstSize = Align(first.Length, 4);
        byte[] notes = new byte[firstSize + second.Length];
        first.CopyTo(notes, 0);
        second.CopyTo(notes, firstSize);
        int sectionOffset = Align(noteOffset + notes.Length, 4);
        var data = new byte[sectionOffset + 3 * shEnt];
        data[0]=0x7F; data[1]=(byte)'E'; data[2]=(byte)'L'; data[3]=(byte)'F';
        data[4]=1; data[5]=1; data[6]=1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 190);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEnt);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 3);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);
        shstr.CopyTo(data, shstrOffset);
        notes.CopyTo(data, noteOffset);
        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEnt;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEnt;
        WriteSection32(data, sh, 11, 7, 0, noteOffset, notes.Length, 4, 0);
        return data;
    }

    internal static byte[] BuildElf32Reloc()
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0.text\0.rela.text\0.symtab\0.strtab\0");
        byte[] strtab = Encoding.ASCII.GetBytes("\0f\0.text\0g\0");
        byte[] text = new byte[48];
        byte[] symtab = new byte[5 * 16];
        byte[] rela = new byte[3 * 12];

        void Symbol(int index, uint name, byte info, ushort section, uint value, uint size)
        {
            int o = index * 16;
            BinaryPrimitives.WriteUInt32LittleEndian(symtab.AsSpan(o), name);
            BinaryPrimitives.WriteUInt32LittleEndian(symtab.AsSpan(o + 4), value);
            BinaryPrimitives.WriteUInt32LittleEndian(symtab.AsSpan(o + 8), size);
            symtab[o + 12] = info;
            symtab[o + 13] = 0;
            BinaryPrimitives.WriteUInt16LittleEndian(symtab.AsSpan(o + 14), section);
        }
        Symbol(1, 1, 0x12, 2, 0, 16);
        Symbol(2, 3, 0x03, 2, 0, 0);
        Symbol(3, 1, 0x12, 2, 16, 16);
        Symbol(4, 9, 0x12, 2, 32, 16);

        void Relocation(int index, uint offset, uint symbol, byte type, int addend)
        {
            int e = index * 12;
            BinaryPrimitives.WriteUInt32LittleEndian(rela.AsSpan(e), offset);
            BinaryPrimitives.WriteUInt32LittleEndian(rela.AsSpan(e + 4), (symbol << 8) | type);
            BinaryPrimitives.WriteInt32LittleEndian(rela.AsSpan(e + 8), addend);
        }
        Relocation(0, 0, 1, 56, 0x20);
        Relocation(1, 16, 3, 57, 0x20);
        Relocation(2, 32, 4, 58, 0);

        const int headerSize = 52;
        const int shEnt = 40;
        int shstrOffset = headerSize;
        int strtabOffset = Align(shstrOffset + shstr.Length, 4);
        int symtabOffset = Align(strtabOffset + strtab.Length, 4);
        int textOffset = Align(symtabOffset + symtab.Length, 16);
        int relaOffset = Align(textOffset + text.Length, 4);
        int sectionOffset = Align(relaOffset + rela.Length, 4);

        var data = new byte[sectionOffset + 6 * shEnt];
        data[0]=0x7F; data[1]=(byte)'E'; data[2]=(byte)'L'; data[3]=(byte)'F';
        data[4]=1; data[5]=1; data[6]=1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEnt);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 6);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);

        shstr.CopyTo(data, shstrOffset);
        strtab.CopyTo(data, strtabOffset);
        symtab.CopyTo(data, symtabOffset);
        text.CopyTo(data, textOffset);
        rela.CopyTo(data, relaOffset);

        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEnt;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEnt;
        WriteSection32(data, sh, 11, 1, 0x6, textOffset, text.Length, 16, 0); sh += shEnt;
        WriteSection32(data, sh, 17, 4, 0, relaOffset, rela.Length, 4, 4, 2, 12); sh += shEnt;
        WriteSection32(data, sh, 29, 2, 0, symtabOffset, symtab.Length, 4, 5, 1, 16); sh += shEnt;
        WriteSection32(data, sh, 38, 3, 0, strtabOffset, strtab.Length, 1, 0);
        return data;
    }

    internal static byte[] BuildElf32Brx()
    {
        byte[] shstr = Encoding.ASCII.GetBytes("\0.shstrtab\0.text.foo\0.nv.info.foo\0");
        byte[] text = new byte[0x200];
        byte[] info = new byte[4 + 24 + 8];
        info[0] = 4; info[1] = 52;
        BinaryPrimitives.WriteUInt16LittleEndian(info.AsSpan(2), 24);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(4), 0x10);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(8), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(12), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(16), 0x100);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(20), 0x120);
        BinaryPrimitives.WriteUInt32LittleEndian(info.AsSpan(24), 0x140);
        info[28] = 4; info[29] = 102;
        BinaryPrimitives.WriteUInt16LittleEndian(info.AsSpan(30), 4);

        const int headerSize = 52;
        const int shEnt = 40;
        int shstrOffset = headerSize;
        int textOffset = Align(shstrOffset + shstr.Length, 16);
        int infoOffset = Align(textOffset + text.Length, 4);
        int sectionOffset = Align(infoOffset + info.Length, 4);

        var data = new byte[sectionOffset + 4 * shEnt];
        data[0]=0x7F; data[1]=(byte)'E'; data[2]=(byte)'L'; data[3]=(byte)'F';
        data[4]=1; data[5]=1; data[6]=1;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(32), (uint)sectionOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(46), shEnt);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(48), 4);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(50), 1);

        shstr.CopyTo(data, shstrOffset);
        text.CopyTo(data, textOffset);
        info.CopyTo(data, infoOffset);

        int sh = sectionOffset;
        WriteSection32(data, sh, 0, 0, 0, 0, 0, 0, 0); sh += shEnt;
        WriteSection32(data, sh, 1, 3, 0, shstrOffset, shstr.Length, 1, 0); sh += shEnt;
        WriteSection32(data, sh, 11, 1, 0x6, textOffset, text.Length, 16, 0); sh += shEnt;
        WriteSection32(data, sh, 21, 1, 0, infoOffset, info.Length, 4, 0);
        return data;
    }

    internal static int Align(int value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;
}
