; ModuleID = "vector_add$1"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

@"_ZN08NumbaEnv8__main__10vector_addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw1dRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE" = common global i8* null
define i32 @"_ZN8__main__10vector_addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw1dRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE"(i8** %".ret", i8* %"arg.a.0", i8* %"arg.a.1", i64 %"arg.a.2", i64 %"arg.a.3", double* %"arg.a.4", i64 %"arg.a.5.0", i64 %"arg.a.6.0", i8* %"arg.b.0", i8* %"arg.b.1", i64 %"arg.b.2", i64 %"arg.b.3", double* %"arg.b.4", i64 %"arg.b.5.0", i64 %"arg.b.6.0", i8* %"arg.c.0", i8* %"arg.c.1", i64 %"arg.c.2", i64 %"arg.c.3", double* %"arg.c.4", i64 %"arg.c.5.0", i64 %"arg.c.6.0")
{
entry:
  %"inserted.meminfo" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} undef, i8* %"arg.a.0", 0
  %"inserted.parent" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.meminfo", i8* %"arg.a.1", 1
  %"inserted.nitems" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.parent", i64 %"arg.a.2", 2
  %"inserted.itemsize" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.nitems", i64 %"arg.a.3", 3
  %"inserted.data" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.itemsize", double* %"arg.a.4", 4
  %".24" = insertvalue [1 x i64] undef, i64 %"arg.a.5.0", 0
  %"inserted.shape" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.data", [1 x i64] %".24", 5
  %".25" = insertvalue [1 x i64] undef, i64 %"arg.a.6.0", 0
  %"inserted.strides" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.shape", [1 x i64] %".25", 6
  %"inserted.meminfo.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} undef, i8* %"arg.b.0", 0
  %"inserted.parent.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.meminfo.1", i8* %"arg.b.1", 1
  %"inserted.nitems.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.parent.1", i64 %"arg.b.2", 2
  %"inserted.itemsize.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.nitems.1", i64 %"arg.b.3", 3
  %"inserted.data.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.itemsize.1", double* %"arg.b.4", 4
  %".26" = insertvalue [1 x i64] undef, i64 %"arg.b.5.0", 0
  %"inserted.shape.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.data.1", [1 x i64] %".26", 5
  %".27" = insertvalue [1 x i64] undef, i64 %"arg.b.6.0", 0
  %"inserted.strides.1" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.shape.1", [1 x i64] %".27", 6
  %"inserted.meminfo.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} undef, i8* %"arg.c.0", 0
  %"inserted.parent.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.meminfo.2", i8* %"arg.c.1", 1
  %"inserted.nitems.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.parent.2", i64 %"arg.c.2", 2
  %"inserted.itemsize.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.nitems.2", i64 %"arg.c.3", 3
  %"inserted.data.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.itemsize.2", double* %"arg.c.4", 4
  %".28" = insertvalue [1 x i64] undef, i64 %"arg.c.5.0", 0
  %"inserted.shape.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.data.2", [1 x i64] %".28", 5
  %".29" = insertvalue [1 x i64] undef, i64 %"arg.c.6.0", 0
  %"inserted.strides.2" = insertvalue {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.shape.2", [1 x i64] %".29", 6
  %"a" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  %"b" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  %"c" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  %"i" = alloca i64
  store i64 0, i64* %"i"
  %".49" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".49"
  %".60" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60"
  %".94" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94"
  %".129" = alloca {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129"
  br label %"B0"
B0:
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.strides", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.strides.1", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %"inserted.strides.2", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  %".36" = trunc i64 1 to i32
  %".37" = call i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()
  %".38" = sext i32 %".37" to i64
  %".39" = call i32 @"llvm.nvvm.read.ptx.sreg.ntid.x"()
  %".40" = sext i32 %".39" to i64
  %".41" = call i32 @"llvm.nvvm.read.ptx.sreg.ctaid.x"()
  %".42" = sext i32 %".41" to i64
  %".43" = mul i64 %".40", %".42"
  %".44" = add i64 %".43", %".38"
  %".46" = load i64, i64* %"i"
  store i64 %".44", i64* %"i"
  %".48" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".49"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %".48", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".49"
  %".53" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".49", i32 0, i32 2
  %".54" = load i64, i64* %".53"
  %".55" = load i64, i64* %"i"
  %".56" = icmp slt i64 %".55", %".54"
  br i1 %".56", label %"B76", label %"B106"
B76:
  %".58" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  %".59" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %".58", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60"
  %".64" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60", i32 0, i32 5
  %".65" = getelementptr inbounds [1 x i64], [1 x i64]* %".64", i32 0, i32 0
  %".66" = load i64, i64* %".65", !range !1
  %".67" = insertvalue [1 x i64] undef, i64 %".66", 0
  %".68" = extractvalue [1 x i64] %".67", 0
  %".69" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60", i32 0, i32 6
  %".70" = load [1 x i64], [1 x i64]* %".69"
  %".71" = extractvalue [1 x i64] %".70", 0
  %".72" = icmp slt i64 %".59", 0
  %".73" = add i64 %".59", %".68"
  %".74" = select  i1 %".72", i64 %".73", i64 %".59"
  %".75" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60", i32 0, i32 5
  %".76" = getelementptr inbounds [1 x i64], [1 x i64]* %".75", i32 0, i32 0
  %".77" = load i64, i64* %".76", !range !1
  %".78" = insertvalue [1 x i64] undef, i64 %".77", 0
  %".79" = extractvalue [1 x i64] %".78", 0
  %".80" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60", i32 0, i32 6
  %".81" = load [1 x i64], [1 x i64]* %".80"
  %".82" = extractvalue [1 x i64] %".81", 0
  %".83" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".60", i32 0, i32 4
  %".84" = load double*, double** %".83"
  %".85" = mul i64 %".82", %".74"
  %".86" = bitcast double* %".84" to i8*
  %".87" = getelementptr i8, i8* %".86", i64 %".85"
  %".88" = bitcast i8* %".87" to double*
  %".89" = load double, double* %".88"
  %".90" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  %".92" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  %".93" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %".92", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94"
  %".98" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94", i32 0, i32 5
  %".99" = getelementptr inbounds [1 x i64], [1 x i64]* %".98", i32 0, i32 0
  %".100" = load i64, i64* %".99", !range !1
  %".101" = insertvalue [1 x i64] undef, i64 %".100", 0
  %".102" = extractvalue [1 x i64] %".101", 0
  %".103" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94", i32 0, i32 6
  %".104" = load [1 x i64], [1 x i64]* %".103"
  %".105" = extractvalue [1 x i64] %".104", 0
  %".106" = icmp slt i64 %".93", 0
  %".107" = add i64 %".93", %".102"
  %".108" = select  i1 %".106", i64 %".107", i64 %".93"
  %".109" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94", i32 0, i32 5
  %".110" = getelementptr inbounds [1 x i64], [1 x i64]* %".109", i32 0, i32 0
  %".111" = load i64, i64* %".110", !range !1
  %".112" = insertvalue [1 x i64] undef, i64 %".111", 0
  %".113" = extractvalue [1 x i64] %".112", 0
  %".114" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94", i32 0, i32 6
  %".115" = load [1 x i64], [1 x i64]* %".114"
  %".116" = extractvalue [1 x i64] %".115", 0
  %".117" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".94", i32 0, i32 4
  %".118" = load double*, double** %".117"
  %".119" = mul i64 %".116", %".108"
  %".120" = bitcast double* %".118" to i8*
  %".121" = getelementptr i8, i8* %".120", i64 %".119"
  %".122" = bitcast i8* %".121" to double*
  %".123" = load double, double* %".122"
  %".124" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  %".126" = fadd double %".89", %".123"
  %".127" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  %".128" = load i64, i64* %"i"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} %".127", {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129"
  %".133" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129", i32 0, i32 5
  %".134" = getelementptr inbounds [1 x i64], [1 x i64]* %".133", i32 0, i32 0
  %".135" = load i64, i64* %".134", !range !1
  %".136" = insertvalue [1 x i64] undef, i64 %".135", 0
  %".137" = extractvalue [1 x i64] %".136", 0
  %".138" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129", i32 0, i32 6
  %".139" = load [1 x i64], [1 x i64]* %".138"
  %".140" = extractvalue [1 x i64] %".139", 0
  %".141" = icmp slt i64 %".128", 0
  %".142" = add i64 %".128", %".137"
  %".143" = select  i1 %".141", i64 %".142", i64 %".128"
  %".144" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129", i32 0, i32 5
  %".145" = getelementptr inbounds [1 x i64], [1 x i64]* %".144", i32 0, i32 0
  %".146" = load i64, i64* %".145", !range !1
  %".147" = insertvalue [1 x i64] undef, i64 %".146", 0
  %".148" = extractvalue [1 x i64] %".147", 0
  %".149" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129", i32 0, i32 6
  %".150" = load [1 x i64], [1 x i64]* %".149"
  %".151" = extractvalue [1 x i64] %".150", 0
  %".152" = getelementptr inbounds {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %".129", i32 0, i32 4
  %".153" = load double*, double** %".152"
  %".154" = mul i64 %".151", %".143"
  %".155" = bitcast double* %".153" to i8*
  %".156" = getelementptr i8, i8* %".155", i64 %".154"
  %".157" = bitcast i8* %".156" to double*
  store double %".126", double* %".157"
  %".159" = load i64, i64* %"i"
  store i64 0, i64* %"i"
  %".161" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  store i8* null, i8** %".ret"
  ret i32 0
B106:
  %".165" = load i64, i64* %"i"
  store i64 0, i64* %"i"
  %".167" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"c"
  %".169" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"b"
  %".171" = load {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  store {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]} zeroinitializer, {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}* %"a"
  store i8* null, i8** %".ret"
  ret i32 0
}

declare i32 @"llvm.nvvm.read.ptx.sreg.tid.x"()

declare i32 @"llvm.nvvm.read.ptx.sreg.ntid.x"()

declare i32 @"llvm.nvvm.read.ptx.sreg.ctaid.x"()

!nvvmir.version = !{ !0 }
!0 = !{ i32 2, i32 0, i32 3, i32 2 }
!1 = !{ i64 0, i64 9223372036854775807 }