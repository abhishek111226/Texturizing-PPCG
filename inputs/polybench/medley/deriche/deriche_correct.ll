; ModuleID = 'deriche_correct.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@stderr = external global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [23 x i8] c"==BEGIN DUMP_ARRAYS==\0A\00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"begin dump: %s\00", align 1
@.str.3 = private unnamed_addr constant [7 x i8] c"imgOut\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.5 = private unnamed_addr constant [7 x i8] c"%0.2f \00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"\0Aend   dump: %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [23 x i8] c"==END   DUMP_ARRAYS==\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %w = alloca i32, align 4
  %h = alloca i32, align 4
  %alpha = alloca float, align 4
  %imgIn = alloca [4096 x [2160 x float]]*, align 8
  %imgOut = alloca [4096 x [2160 x float]]*, align 8
  %y1 = alloca [4096 x [2160 x float]]*, align 8
  %y2 = alloca [4096 x [2160 x float]]*, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  store i32 4096, i32* %w, align 4
  store i32 2160, i32* %h, align 4
  %call = call i8* @polybench_alloc_data(i64 8847360, i32 4)
  %0 = bitcast i8* %call to [4096 x [2160 x float]]*
  store [4096 x [2160 x float]]* %0, [4096 x [2160 x float]]** %imgIn, align 8
  %call1 = call i8* @polybench_alloc_data(i64 8847360, i32 4)
  %1 = bitcast i8* %call1 to [4096 x [2160 x float]]*
  store [4096 x [2160 x float]]* %1, [4096 x [2160 x float]]** %imgOut, align 8
  %call2 = call i8* @polybench_alloc_data(i64 8847360, i32 4)
  %2 = bitcast i8* %call2 to [4096 x [2160 x float]]*
  store [4096 x [2160 x float]]* %2, [4096 x [2160 x float]]** %y1, align 8
  %call3 = call i8* @polybench_alloc_data(i64 8847360, i32 4)
  %3 = bitcast i8* %call3 to [4096 x [2160 x float]]*
  store [4096 x [2160 x float]]* %3, [4096 x [2160 x float]]** %y2, align 8
  %4 = load i32, i32* %w, align 4
  %5 = load i32, i32* %h, align 4
  %6 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgIn, align 8
  %arraydecay = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %6, i32 0, i32 0
  %7 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgOut, align 8
  %arraydecay4 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %7, i32 0, i32 0
  call void @init_array(i32 %4, i32 %5, float* %alpha, [2160 x float]* %arraydecay, [2160 x float]* %arraydecay4)
  call void (...) @polybench_timer_start()
  %8 = load i32, i32* %w, align 4
  %9 = load i32, i32* %h, align 4
  %10 = load float, float* %alpha, align 4
  %11 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgIn, align 8
  %arraydecay5 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %11, i32 0, i32 0
  %12 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgOut, align 8
  %arraydecay6 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %12, i32 0, i32 0
  %13 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %y1, align 8
  %arraydecay7 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %13, i32 0, i32 0
  %14 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %y2, align 8
  %arraydecay8 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %14, i32 0, i32 0
  call void @kernel_deriche(i32 %8, i32 %9, float %10, [2160 x float]* %arraydecay5, [2160 x float]* %arraydecay6, [2160 x float]* %arraydecay7, [2160 x float]* %arraydecay8)
  call void (...) @polybench_timer_stop()
  call void (...) @polybench_timer_print()
  %15 = load i32, i32* %argc.addr, align 4
  %cmp = icmp sgt i32 %15, 42
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %16 = load i8**, i8*** %argv.addr, align 8
  %arrayidx = getelementptr inbounds i8*, i8** %16, i64 0
  %17 = load i8*, i8** %arrayidx, align 8
  %call9 = call i32 @strcmp(i8* %17, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0)) #4
  %tobool = icmp ne i32 %call9, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %land.lhs.true
  %18 = load i32, i32* %w, align 4
  %19 = load i32, i32* %h, align 4
  %20 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgOut, align 8
  %arraydecay10 = getelementptr inbounds [4096 x [2160 x float]], [4096 x [2160 x float]]* %20, i32 0, i32 0
  call void @print_array(i32 %18, i32 %19, [2160 x float]* %arraydecay10)
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  %21 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgIn, align 8
  %22 = bitcast [4096 x [2160 x float]]* %21 to i8*
  call void @free(i8* %22) #5
  %23 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %imgOut, align 8
  %24 = bitcast [4096 x [2160 x float]]* %23 to i8*
  call void @free(i8* %24) #5
  %25 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %y1, align 8
  %26 = bitcast [4096 x [2160 x float]]* %25 to i8*
  call void @free(i8* %26) #5
  %27 = load [4096 x [2160 x float]]*, [4096 x [2160 x float]]** %y2, align 8
  %28 = bitcast [4096 x [2160 x float]]* %27 to i8*
  call void @free(i8* %28) #5
  ret i32 0
}

declare i8* @polybench_alloc_data(i64, i32) #1

; Function Attrs: nounwind uwtable
define internal void @init_array(i32 %w, i32 %h, float* %alpha, [2160 x float]* %imgIn, [2160 x float]* %imgOut) #0 {
entry:
  %w.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %alpha.addr = alloca float*, align 8
  %imgIn.addr = alloca [2160 x float]*, align 8
  %imgOut.addr = alloca [2160 x float]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %w, i32* %w.addr, align 4
  store i32 %h, i32* %h.addr, align 4
  store float* %alpha, float** %alpha.addr, align 8
  store [2160 x float]* %imgIn, [2160 x float]** %imgIn.addr, align 8
  store [2160 x float]* %imgOut, [2160 x float]** %imgOut.addr, align 8
  %0 = load float*, float** %alpha.addr, align 8
  store float 2.500000e-01, float* %0, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.7, %entry
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %w.addr, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end.9

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %3 = load i32, i32* %j, align 4
  %4 = load i32, i32* %h.addr, align 4
  %cmp2 = icmp slt i32 %3, %4
  br i1 %cmp2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  %5 = load i32, i32* %i, align 4
  %mul = mul nsw i32 313, %5
  %6 = load i32, i32* %j, align 4
  %mul4 = mul nsw i32 991, %6
  %add = add nsw i32 %mul, %mul4
  %rem = srem i32 %add, 65536
  %conv = sitofp i32 %rem to float
  %div = fdiv float %conv, 6.553500e+04
  %7 = load i32, i32* %j, align 4
  %idxprom = sext i32 %7 to i64
  %8 = load i32, i32* %i, align 4
  %idxprom5 = sext i32 %8 to i64
  %9 = load [2160 x float]*, [2160 x float]** %imgIn.addr, align 8
  %arrayidx = getelementptr inbounds [2160 x float], [2160 x float]* %9, i64 %idxprom5
  %arrayidx6 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx, i32 0, i64 %idxprom
  store float %div, float* %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.3
  %10 = load i32, i32* %j, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.7

for.inc.7:                                        ; preds = %for.end
  %11 = load i32, i32* %i, align 4
  %inc8 = add nsw i32 %11, 1
  store i32 %inc8, i32* %i, align 4
  br label %for.cond

for.end.9:                                        ; preds = %for.cond
  ret void
}

declare void @polybench_timer_start(...) #1

; Function Attrs: nounwind uwtable
define internal void @kernel_deriche(i32 %w, i32 %h, float %alpha, [2160 x float]* %imgIn, [2160 x float]* %imgOut, [2160 x float]* %y1, [2160 x float]* %y2) #0 {
entry:
  %w.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %alpha.addr = alloca float, align 4
  %imgIn.addr = alloca [2160 x float]*, align 8
  %imgOut.addr = alloca [2160 x float]*, align 8
  %y1.addr = alloca [2160 x float]*, align 8
  %y2.addr = alloca [2160 x float]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %xm1 = alloca float, align 4
  %tm1 = alloca float, align 4
  %ym1 = alloca float, align 4
  %ym2 = alloca float, align 4
  %xp1 = alloca float, align 4
  %xp2 = alloca float, align 4
  %tp1 = alloca float, align 4
  %tp2 = alloca float, align 4
  %yp1 = alloca float, align 4
  %yp2 = alloca float, align 4
  %k = alloca float, align 4
  %a1 = alloca float, align 4
  %a2 = alloca float, align 4
  %a3 = alloca float, align 4
  %a4 = alloca float, align 4
  %a5 = alloca float, align 4
  %a6 = alloca float, align 4
  %a7 = alloca float, align 4
  %a8 = alloca float, align 4
  %b1 = alloca float, align 4
  %b2 = alloca float, align 4
  %c1 = alloca float, align 4
  %c2 = alloca float, align 4
  store i32 %w, i32* %w.addr, align 4
  store i32 %h, i32* %h.addr, align 4
  store float %alpha, float* %alpha.addr, align 4
  store [2160 x float]* %imgIn, [2160 x float]** %imgIn.addr, align 8
  store [2160 x float]* %imgOut, [2160 x float]** %imgOut.addr, align 8
  store [2160 x float]* %y1, [2160 x float]** %y1.addr, align 8
  store [2160 x float]* %y2, [2160 x float]** %y2.addr, align 8
  %0 = load float, float* %alpha.addr, align 4
  %sub = fsub float -0.000000e+00, %0
  %call = call float @expf(float %sub) #5
  %sub1 = fsub float 1.000000e+00, %call
  %1 = load float, float* %alpha.addr, align 4
  %sub2 = fsub float -0.000000e+00, %1
  %call3 = call float @expf(float %sub2) #5
  %sub4 = fsub float 1.000000e+00, %call3
  %mul = fmul float %sub1, %sub4
  %2 = load float, float* %alpha.addr, align 4
  %mul5 = fmul float 2.000000e+00, %2
  %3 = load float, float* %alpha.addr, align 4
  %sub6 = fsub float -0.000000e+00, %3
  %call7 = call float @expf(float %sub6) #5
  %mul8 = fmul float %mul5, %call7
  %add = fadd float 1.000000e+00, %mul8
  %4 = load float, float* %alpha.addr, align 4
  %mul9 = fmul float 2.000000e+00, %4
  %call10 = call float @expf(float %mul9) #5
  %sub11 = fsub float %add, %call10
  %div = fdiv float %mul, %sub11
  store float %div, float* %k, align 4
  %5 = load float, float* %k, align 4
  store float %5, float* %a5, align 4
  store float %5, float* %a1, align 4
  %6 = load float, float* %k, align 4
  %7 = load float, float* %alpha.addr, align 4
  %sub12 = fsub float -0.000000e+00, %7
  %call13 = call float @expf(float %sub12) #5
  %mul14 = fmul float %6, %call13
  %8 = load float, float* %alpha.addr, align 4
  %sub15 = fsub float %8, 1.000000e+00
  %mul16 = fmul float %mul14, %sub15
  store float %mul16, float* %a6, align 4
  store float %mul16, float* %a2, align 4
  %9 = load float, float* %k, align 4
  %10 = load float, float* %alpha.addr, align 4
  %sub17 = fsub float -0.000000e+00, %10
  %call18 = call float @expf(float %sub17) #5
  %mul19 = fmul float %9, %call18
  %11 = load float, float* %alpha.addr, align 4
  %add20 = fadd float %11, 1.000000e+00
  %mul21 = fmul float %mul19, %add20
  store float %mul21, float* %a7, align 4
  store float %mul21, float* %a3, align 4
  %12 = load float, float* %k, align 4
  %sub22 = fsub float -0.000000e+00, %12
  %13 = load float, float* %alpha.addr, align 4
  %mul23 = fmul float -2.000000e+00, %13
  %call24 = call float @expf(float %mul23) #5
  %mul25 = fmul float %sub22, %call24
  store float %mul25, float* %a8, align 4
  store float %mul25, float* %a4, align 4
  %14 = load float, float* %alpha.addr, align 4
  %sub26 = fsub float -0.000000e+00, %14
  %call27 = call float @powf(float 2.000000e+00, float %sub26) #5
  store float %call27, float* %b1, align 4
  %15 = load float, float* %alpha.addr, align 4
  %mul28 = fmul float -2.000000e+00, %15
  %call29 = call float @expf(float %mul28) #5
  %sub30 = fsub float -0.000000e+00, %call29
  store float %sub30, float* %b2, align 4
  store float 1.000000e+00, float* %c2, align 4
  store float 1.000000e+00, float* %c1, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.55, %entry
  %16 = load i32, i32* %i, align 4
  %17 = load i32, i32* %w.addr, align 4
  %cmp = icmp slt i32 %16, %17
  br i1 %cmp, label %for.body, label %for.end.57

for.body:                                         ; preds = %for.cond
  store float 0.000000e+00, float* %ym1, align 4
  store float 0.000000e+00, float* %ym2, align 4
  store float 0.000000e+00, float* %xm1, align 4
  store i32 0, i32* %j, align 4
  br label %for.cond.31

for.cond.31:                                      ; preds = %for.inc, %for.body
  %18 = load i32, i32* %j, align 4
  %19 = load i32, i32* %h.addr, align 4
  %cmp32 = icmp slt i32 %18, %19
  br i1 %cmp32, label %for.body.33, label %for.end

for.body.33:                                      ; preds = %for.cond.31
  %20 = load float, float* %a1, align 4
  %21 = load i32, i32* %j, align 4
  %idxprom = sext i32 %21 to i64
  %22 = load i32, i32* %i, align 4
  %idxprom34 = sext i32 %22 to i64
  %23 = load [2160 x float]*, [2160 x float]** %imgIn.addr, align 8
  %arrayidx = getelementptr inbounds [2160 x float], [2160 x float]* %23, i64 %idxprom34
  %arrayidx35 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx, i32 0, i64 %idxprom
  %24 = load float, float* %arrayidx35, align 4
  %mul36 = fmul float %20, %24
  %25 = load float, float* %a2, align 4
  %26 = load float, float* %xm1, align 4
  %mul37 = fmul float %25, %26
  %add38 = fadd float %mul36, %mul37
  %27 = load float, float* %b1, align 4
  %28 = load float, float* %ym1, align 4
  %mul39 = fmul float %27, %28
  %add40 = fadd float %add38, %mul39
  %29 = load float, float* %b2, align 4
  %30 = load float, float* %ym2, align 4
  %mul41 = fmul float %29, %30
  %add42 = fadd float %add40, %mul41
  %31 = load i32, i32* %j, align 4
  %idxprom43 = sext i32 %31 to i64
  %32 = load i32, i32* %i, align 4
  %idxprom44 = sext i32 %32 to i64
  %33 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx45 = getelementptr inbounds [2160 x float], [2160 x float]* %33, i64 %idxprom44
  %arrayidx46 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx45, i32 0, i64 %idxprom43
  store float %add42, float* %arrayidx46, align 4
  %34 = load i32, i32* %j, align 4
  %idxprom47 = sext i32 %34 to i64
  %35 = load i32, i32* %i, align 4
  %idxprom48 = sext i32 %35 to i64
  %36 = load [2160 x float]*, [2160 x float]** %imgIn.addr, align 8
  %arrayidx49 = getelementptr inbounds [2160 x float], [2160 x float]* %36, i64 %idxprom48
  %arrayidx50 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx49, i32 0, i64 %idxprom47
  %37 = load float, float* %arrayidx50, align 4
  store float %37, float* %xm1, align 4
  %38 = load float, float* %ym1, align 4
  store float %38, float* %ym2, align 4
  %39 = load i32, i32* %j, align 4
  %idxprom51 = sext i32 %39 to i64
  %40 = load i32, i32* %i, align 4
  %idxprom52 = sext i32 %40 to i64
  %41 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx53 = getelementptr inbounds [2160 x float], [2160 x float]* %41, i64 %idxprom52
  %arrayidx54 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx53, i32 0, i64 %idxprom51
  %42 = load float, float* %arrayidx54, align 4
  store float %42, float* %ym1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.33
  %43 = load i32, i32* %j, align 4
  %inc = add nsw i32 %43, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond.31

for.end:                                          ; preds = %for.cond.31
  br label %for.inc.55

for.inc.55:                                       ; preds = %for.end
  %44 = load i32, i32* %i, align 4
  %inc56 = add nsw i32 %44, 1
  store i32 %inc56, i32* %i, align 4
  br label %for.cond

for.end.57:                                       ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond.58

for.cond.58:                                      ; preds = %for.inc.86, %for.end.57
  %45 = load i32, i32* %i, align 4
  %46 = load i32, i32* %w.addr, align 4
  %cmp59 = icmp slt i32 %45, %46
  br i1 %cmp59, label %for.body.60, label %for.end.88

for.body.60:                                      ; preds = %for.cond.58
  store float 0.000000e+00, float* %yp1, align 4
  store float 0.000000e+00, float* %yp2, align 4
  store float 0.000000e+00, float* %xp1, align 4
  store float 0.000000e+00, float* %xp2, align 4
  %47 = load i32, i32* %h.addr, align 4
  %sub61 = sub nsw i32 %47, 1
  store i32 %sub61, i32* %j, align 4
  br label %for.cond.62

for.cond.62:                                      ; preds = %for.inc.84, %for.body.60
  %48 = load i32, i32* %j, align 4
  %cmp63 = icmp sge i32 %48, 0
  br i1 %cmp63, label %for.body.64, label %for.end.85

for.body.64:                                      ; preds = %for.cond.62
  %49 = load float, float* %a3, align 4
  %50 = load float, float* %xp1, align 4
  %mul65 = fmul float %49, %50
  %51 = load float, float* %a4, align 4
  %52 = load float, float* %xp2, align 4
  %mul66 = fmul float %51, %52
  %add67 = fadd float %mul65, %mul66
  %53 = load float, float* %b1, align 4
  %54 = load float, float* %yp1, align 4
  %mul68 = fmul float %53, %54
  %add69 = fadd float %add67, %mul68
  %55 = load float, float* %b2, align 4
  %56 = load float, float* %yp2, align 4
  %mul70 = fmul float %55, %56
  %add71 = fadd float %add69, %mul70
  %57 = load i32, i32* %j, align 4
  %idxprom72 = sext i32 %57 to i64
  %58 = load i32, i32* %i, align 4
  %idxprom73 = sext i32 %58 to i64
  %59 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx74 = getelementptr inbounds [2160 x float], [2160 x float]* %59, i64 %idxprom73
  %arrayidx75 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx74, i32 0, i64 %idxprom72
  store float %add71, float* %arrayidx75, align 4
  %60 = load float, float* %xp1, align 4
  store float %60, float* %xp2, align 4
  %61 = load i32, i32* %j, align 4
  %idxprom76 = sext i32 %61 to i64
  %62 = load i32, i32* %i, align 4
  %idxprom77 = sext i32 %62 to i64
  %63 = load [2160 x float]*, [2160 x float]** %imgIn.addr, align 8
  %arrayidx78 = getelementptr inbounds [2160 x float], [2160 x float]* %63, i64 %idxprom77
  %arrayidx79 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx78, i32 0, i64 %idxprom76
  %64 = load float, float* %arrayidx79, align 4
  store float %64, float* %xp1, align 4
  %65 = load float, float* %yp1, align 4
  store float %65, float* %yp2, align 4
  %66 = load i32, i32* %j, align 4
  %idxprom80 = sext i32 %66 to i64
  %67 = load i32, i32* %i, align 4
  %idxprom81 = sext i32 %67 to i64
  %68 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx82 = getelementptr inbounds [2160 x float], [2160 x float]* %68, i64 %idxprom81
  %arrayidx83 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx82, i32 0, i64 %idxprom80
  %69 = load float, float* %arrayidx83, align 4
  store float %69, float* %yp1, align 4
  br label %for.inc.84

for.inc.84:                                       ; preds = %for.body.64
  %70 = load i32, i32* %j, align 4
  %dec = add nsw i32 %70, -1
  store i32 %dec, i32* %j, align 4
  br label %for.cond.62

for.end.85:                                       ; preds = %for.cond.62
  br label %for.inc.86

for.inc.86:                                       ; preds = %for.end.85
  %71 = load i32, i32* %i, align 4
  %inc87 = add nsw i32 %71, 1
  store i32 %inc87, i32* %i, align 4
  br label %for.cond.58

for.end.88:                                       ; preds = %for.cond.58
  store i32 0, i32* %i, align 4
  br label %for.cond.89

for.cond.89:                                      ; preds = %for.inc.112, %for.end.88
  %72 = load i32, i32* %i, align 4
  %73 = load i32, i32* %w.addr, align 4
  %cmp90 = icmp slt i32 %72, %73
  br i1 %cmp90, label %for.body.91, label %for.end.114

for.body.91:                                      ; preds = %for.cond.89
  store i32 0, i32* %j, align 4
  br label %for.cond.92

for.cond.92:                                      ; preds = %for.inc.109, %for.body.91
  %74 = load i32, i32* %j, align 4
  %75 = load i32, i32* %h.addr, align 4
  %cmp93 = icmp slt i32 %74, %75
  br i1 %cmp93, label %for.body.94, label %for.end.111

for.body.94:                                      ; preds = %for.cond.92
  %76 = load float, float* %c1, align 4
  %77 = load i32, i32* %j, align 4
  %idxprom95 = sext i32 %77 to i64
  %78 = load i32, i32* %i, align 4
  %idxprom96 = sext i32 %78 to i64
  %79 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx97 = getelementptr inbounds [2160 x float], [2160 x float]* %79, i64 %idxprom96
  %arrayidx98 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx97, i32 0, i64 %idxprom95
  %80 = load float, float* %arrayidx98, align 4
  %81 = load i32, i32* %j, align 4
  %idxprom99 = sext i32 %81 to i64
  %82 = load i32, i32* %i, align 4
  %idxprom100 = sext i32 %82 to i64
  %83 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx101 = getelementptr inbounds [2160 x float], [2160 x float]* %83, i64 %idxprom100
  %arrayidx102 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx101, i32 0, i64 %idxprom99
  %84 = load float, float* %arrayidx102, align 4
  %add103 = fadd float %80, %84
  %mul104 = fmul float %76, %add103
  %85 = load i32, i32* %j, align 4
  %idxprom105 = sext i32 %85 to i64
  %86 = load i32, i32* %i, align 4
  %idxprom106 = sext i32 %86 to i64
  %87 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx107 = getelementptr inbounds [2160 x float], [2160 x float]* %87, i64 %idxprom106
  %arrayidx108 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx107, i32 0, i64 %idxprom105
  store float %mul104, float* %arrayidx108, align 4
  br label %for.inc.109

for.inc.109:                                      ; preds = %for.body.94
  %88 = load i32, i32* %j, align 4
  %inc110 = add nsw i32 %88, 1
  store i32 %inc110, i32* %j, align 4
  br label %for.cond.92

for.end.111:                                      ; preds = %for.cond.92
  br label %for.inc.112

for.inc.112:                                      ; preds = %for.end.111
  %89 = load i32, i32* %i, align 4
  %inc113 = add nsw i32 %89, 1
  store i32 %inc113, i32* %i, align 4
  br label %for.cond.89

for.end.114:                                      ; preds = %for.cond.89
  store i32 0, i32* %j, align 4
  br label %for.cond.115

for.cond.115:                                     ; preds = %for.inc.147, %for.end.114
  %90 = load i32, i32* %j, align 4
  %91 = load i32, i32* %h.addr, align 4
  %cmp116 = icmp slt i32 %90, %91
  br i1 %cmp116, label %for.body.117, label %for.end.149

for.body.117:                                     ; preds = %for.cond.115
  store float 0.000000e+00, float* %tm1, align 4
  store float 0.000000e+00, float* %ym1, align 4
  store float 0.000000e+00, float* %ym2, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond.118

for.cond.118:                                     ; preds = %for.inc.144, %for.body.117
  %92 = load i32, i32* %i, align 4
  %93 = load i32, i32* %w.addr, align 4
  %cmp119 = icmp slt i32 %92, %93
  br i1 %cmp119, label %for.body.120, label %for.end.146

for.body.120:                                     ; preds = %for.cond.118
  %94 = load float, float* %a5, align 4
  %95 = load i32, i32* %j, align 4
  %idxprom121 = sext i32 %95 to i64
  %96 = load i32, i32* %i, align 4
  %idxprom122 = sext i32 %96 to i64
  %97 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx123 = getelementptr inbounds [2160 x float], [2160 x float]* %97, i64 %idxprom122
  %arrayidx124 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx123, i32 0, i64 %idxprom121
  %98 = load float, float* %arrayidx124, align 4
  %mul125 = fmul float %94, %98
  %99 = load float, float* %a6, align 4
  %100 = load float, float* %tm1, align 4
  %mul126 = fmul float %99, %100
  %add127 = fadd float %mul125, %mul126
  %101 = load float, float* %b1, align 4
  %102 = load float, float* %ym1, align 4
  %mul128 = fmul float %101, %102
  %add129 = fadd float %add127, %mul128
  %103 = load float, float* %b2, align 4
  %104 = load float, float* %ym2, align 4
  %mul130 = fmul float %103, %104
  %add131 = fadd float %add129, %mul130
  %105 = load i32, i32* %j, align 4
  %idxprom132 = sext i32 %105 to i64
  %106 = load i32, i32* %i, align 4
  %idxprom133 = sext i32 %106 to i64
  %107 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx134 = getelementptr inbounds [2160 x float], [2160 x float]* %107, i64 %idxprom133
  %arrayidx135 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx134, i32 0, i64 %idxprom132
  store float %add131, float* %arrayidx135, align 4
  %108 = load i32, i32* %j, align 4
  %idxprom136 = sext i32 %108 to i64
  %109 = load i32, i32* %i, align 4
  %idxprom137 = sext i32 %109 to i64
  %110 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx138 = getelementptr inbounds [2160 x float], [2160 x float]* %110, i64 %idxprom137
  %arrayidx139 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx138, i32 0, i64 %idxprom136
  %111 = load float, float* %arrayidx139, align 4
  store float %111, float* %tm1, align 4
  %112 = load float, float* %ym1, align 4
  store float %112, float* %ym2, align 4
  %113 = load i32, i32* %j, align 4
  %idxprom140 = sext i32 %113 to i64
  %114 = load i32, i32* %i, align 4
  %idxprom141 = sext i32 %114 to i64
  %115 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx142 = getelementptr inbounds [2160 x float], [2160 x float]* %115, i64 %idxprom141
  %arrayidx143 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx142, i32 0, i64 %idxprom140
  %116 = load float, float* %arrayidx143, align 4
  store float %116, float* %ym1, align 4
  br label %for.inc.144

for.inc.144:                                      ; preds = %for.body.120
  %117 = load i32, i32* %i, align 4
  %inc145 = add nsw i32 %117, 1
  store i32 %inc145, i32* %i, align 4
  br label %for.cond.118

for.end.146:                                      ; preds = %for.cond.118
  br label %for.inc.147

for.inc.147:                                      ; preds = %for.end.146
  %118 = load i32, i32* %j, align 4
  %inc148 = add nsw i32 %118, 1
  store i32 %inc148, i32* %j, align 4
  br label %for.cond.115

for.end.149:                                      ; preds = %for.cond.115
  store i32 0, i32* %j, align 4
  br label %for.cond.150

for.cond.150:                                     ; preds = %for.inc.179, %for.end.149
  %119 = load i32, i32* %j, align 4
  %120 = load i32, i32* %h.addr, align 4
  %cmp151 = icmp slt i32 %119, %120
  br i1 %cmp151, label %for.body.152, label %for.end.181

for.body.152:                                     ; preds = %for.cond.150
  store float 0.000000e+00, float* %tp1, align 4
  store float 0.000000e+00, float* %tp2, align 4
  store float 0.000000e+00, float* %yp1, align 4
  store float 0.000000e+00, float* %yp2, align 4
  %121 = load i32, i32* %w.addr, align 4
  %sub153 = sub nsw i32 %121, 1
  store i32 %sub153, i32* %i, align 4
  br label %for.cond.154

for.cond.154:                                     ; preds = %for.inc.176, %for.body.152
  %122 = load i32, i32* %i, align 4
  %cmp155 = icmp sge i32 %122, 0
  br i1 %cmp155, label %for.body.156, label %for.end.178

for.body.156:                                     ; preds = %for.cond.154
  %123 = load float, float* %a7, align 4
  %124 = load float, float* %tp1, align 4
  %mul157 = fmul float %123, %124
  %125 = load float, float* %a8, align 4
  %126 = load float, float* %tp2, align 4
  %mul158 = fmul float %125, %126
  %add159 = fadd float %mul157, %mul158
  %127 = load float, float* %b1, align 4
  %128 = load float, float* %yp1, align 4
  %mul160 = fmul float %127, %128
  %add161 = fadd float %add159, %mul160
  %129 = load float, float* %b2, align 4
  %130 = load float, float* %yp2, align 4
  %mul162 = fmul float %129, %130
  %add163 = fadd float %add161, %mul162
  %131 = load i32, i32* %j, align 4
  %idxprom164 = sext i32 %131 to i64
  %132 = load i32, i32* %i, align 4
  %idxprom165 = sext i32 %132 to i64
  %133 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx166 = getelementptr inbounds [2160 x float], [2160 x float]* %133, i64 %idxprom165
  %arrayidx167 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx166, i32 0, i64 %idxprom164
  store float %add163, float* %arrayidx167, align 4
  %134 = load float, float* %tp1, align 4
  store float %134, float* %tp2, align 4
  %135 = load i32, i32* %j, align 4
  %idxprom168 = sext i32 %135 to i64
  %136 = load i32, i32* %i, align 4
  %idxprom169 = sext i32 %136 to i64
  %137 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx170 = getelementptr inbounds [2160 x float], [2160 x float]* %137, i64 %idxprom169
  %arrayidx171 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx170, i32 0, i64 %idxprom168
  %138 = load float, float* %arrayidx171, align 4
  store float %138, float* %tp1, align 4
  %139 = load float, float* %yp1, align 4
  store float %139, float* %yp2, align 4
  %140 = load i32, i32* %j, align 4
  %idxprom172 = sext i32 %140 to i64
  %141 = load i32, i32* %i, align 4
  %idxprom173 = sext i32 %141 to i64
  %142 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx174 = getelementptr inbounds [2160 x float], [2160 x float]* %142, i64 %idxprom173
  %arrayidx175 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx174, i32 0, i64 %idxprom172
  %143 = load float, float* %arrayidx175, align 4
  store float %143, float* %yp1, align 4
  br label %for.inc.176

for.inc.176:                                      ; preds = %for.body.156
  %144 = load i32, i32* %i, align 4
  %dec177 = add nsw i32 %144, -1
  store i32 %dec177, i32* %i, align 4
  br label %for.cond.154

for.end.178:                                      ; preds = %for.cond.154
  br label %for.inc.179

for.inc.179:                                      ; preds = %for.end.178
  %145 = load i32, i32* %j, align 4
  %inc180 = add nsw i32 %145, 1
  store i32 %inc180, i32* %j, align 4
  br label %for.cond.150

for.end.181:                                      ; preds = %for.cond.150
  store i32 0, i32* %i, align 4
  br label %for.cond.182

for.cond.182:                                     ; preds = %for.inc.205, %for.end.181
  %146 = load i32, i32* %i, align 4
  %147 = load i32, i32* %w.addr, align 4
  %cmp183 = icmp slt i32 %146, %147
  br i1 %cmp183, label %for.body.184, label %for.end.207

for.body.184:                                     ; preds = %for.cond.182
  store i32 0, i32* %j, align 4
  br label %for.cond.185

for.cond.185:                                     ; preds = %for.inc.202, %for.body.184
  %148 = load i32, i32* %j, align 4
  %149 = load i32, i32* %h.addr, align 4
  %cmp186 = icmp slt i32 %148, %149
  br i1 %cmp186, label %for.body.187, label %for.end.204

for.body.187:                                     ; preds = %for.cond.185
  %150 = load float, float* %c2, align 4
  %151 = load i32, i32* %j, align 4
  %idxprom188 = sext i32 %151 to i64
  %152 = load i32, i32* %i, align 4
  %idxprom189 = sext i32 %152 to i64
  %153 = load [2160 x float]*, [2160 x float]** %y1.addr, align 8
  %arrayidx190 = getelementptr inbounds [2160 x float], [2160 x float]* %153, i64 %idxprom189
  %arrayidx191 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx190, i32 0, i64 %idxprom188
  %154 = load float, float* %arrayidx191, align 4
  %155 = load i32, i32* %j, align 4
  %idxprom192 = sext i32 %155 to i64
  %156 = load i32, i32* %i, align 4
  %idxprom193 = sext i32 %156 to i64
  %157 = load [2160 x float]*, [2160 x float]** %y2.addr, align 8
  %arrayidx194 = getelementptr inbounds [2160 x float], [2160 x float]* %157, i64 %idxprom193
  %arrayidx195 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx194, i32 0, i64 %idxprom192
  %158 = load float, float* %arrayidx195, align 4
  %add196 = fadd float %154, %158
  %mul197 = fmul float %150, %add196
  %159 = load i32, i32* %j, align 4
  %idxprom198 = sext i32 %159 to i64
  %160 = load i32, i32* %i, align 4
  %idxprom199 = sext i32 %160 to i64
  %161 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx200 = getelementptr inbounds [2160 x float], [2160 x float]* %161, i64 %idxprom199
  %arrayidx201 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx200, i32 0, i64 %idxprom198
  store float %mul197, float* %arrayidx201, align 4
  br label %for.inc.202

for.inc.202:                                      ; preds = %for.body.187
  %162 = load i32, i32* %j, align 4
  %inc203 = add nsw i32 %162, 1
  store i32 %inc203, i32* %j, align 4
  br label %for.cond.185

for.end.204:                                      ; preds = %for.cond.185
  br label %for.inc.205

for.inc.205:                                      ; preds = %for.end.204
  %163 = load i32, i32* %i, align 4
  %inc206 = add nsw i32 %163, 1
  store i32 %inc206, i32* %i, align 4
  br label %for.cond.182

for.end.207:                                      ; preds = %for.cond.182
  ret void
}

declare void @polybench_timer_stop(...) #1

declare void @polybench_timer_print(...) #1

; Function Attrs: nounwind readonly
declare i32 @strcmp(i8*, i8*) #2

; Function Attrs: nounwind uwtable
define internal void @print_array(i32 %w, i32 %h, [2160 x float]* %imgOut) #0 {
entry:
  %w.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %imgOut.addr = alloca [2160 x float]*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %w, i32* %w.addr, align 4
  store i32 %h, i32* %h.addr, align 4
  store [2160 x float]* %imgOut, [2160 x float]** %imgOut.addr, align 8
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0))
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.3, i32 0, i32 0))
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.10, %entry
  %2 = load i32, i32* %i, align 4
  %3 = load i32, i32* %w.addr, align 4
  %cmp = icmp slt i32 %2, %3
  br i1 %cmp, label %for.body, label %for.end.12

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond.2

for.cond.2:                                       ; preds = %for.inc, %for.body
  %4 = load i32, i32* %j, align 4
  %5 = load i32, i32* %h.addr, align 4
  %cmp3 = icmp slt i32 %4, %5
  br i1 %cmp3, label %for.body.4, label %for.end

for.body.4:                                       ; preds = %for.cond.2
  %6 = load i32, i32* %i, align 4
  %7 = load i32, i32* %h.addr, align 4
  %mul = mul nsw i32 %6, %7
  %8 = load i32, i32* %j, align 4
  %add = add nsw i32 %mul, %8
  %rem = srem i32 %add, 20
  %cmp5 = icmp eq i32 %rem, 0
  br i1 %cmp5, label %if.then, label %if.end

if.then:                                          ; preds = %for.body.4
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call6 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %9, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.4, i32 0, i32 0))
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body.4
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %11 = load i32, i32* %j, align 4
  %idxprom = sext i32 %11 to i64
  %12 = load i32, i32* %i, align 4
  %idxprom7 = sext i32 %12 to i64
  %13 = load [2160 x float]*, [2160 x float]** %imgOut.addr, align 8
  %arrayidx = getelementptr inbounds [2160 x float], [2160 x float]* %13, i64 %idxprom7
  %arrayidx8 = getelementptr inbounds [2160 x float], [2160 x float]* %arrayidx, i32 0, i64 %idxprom
  %14 = load float, float* %arrayidx8, align 4
  %conv = fpext float %14 to double
  %call9 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %10, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.5, i32 0, i32 0), double %conv)
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %15 = load i32, i32* %j, align 4
  %inc = add nsw i32 %15, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond.2

for.end:                                          ; preds = %for.cond.2
  br label %for.inc.10

for.inc.10:                                       ; preds = %for.end
  %16 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %16, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond

for.end.12:                                       ; preds = %for.cond
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call13 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %17, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.3, i32 0, i32 0))
  %18 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call14 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %18, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.7, i32 0, i32 0))
  ret void
}

; Function Attrs: nounwind
declare void @free(i8*) #3

; Function Attrs: nounwind
declare float @expf(float) #3

; Function Attrs: nounwind
declare float @powf(float, float) #3

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readonly "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readonly }
attributes #5 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (tags/RELEASE_370/final)"}
