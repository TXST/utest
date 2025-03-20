
.globl conv_1d
conv_1d:
#define in r0
#define out r1
#define size r2

#define wei0 r3
#define wei1 r4
#define wei2 r5
#define wei3 r6
#define wei4 r7
#define wei5 r8
#define wei6 r9
#define wei7 r10
#define wei8 r11

//改成一次拿俩试试
#define in01 r12
#define in23 r13
//out暂时不会弄
#define out0 r15

#define idx r16
#define tmp r17

//void conv_1d(T* in,T* out,int16_t size,T* wei)

//DPU是小端（低字节低地址）
  move  idx,  0
  //in
  //不知道怎么给中间8位赋值，先改大小端吧，或者先不管，只看速度
  lhs   in01,  in,   0      //这里先1字节
  add   in,   in,   2
  //weight ---- 后面试试先加载再移位取出
  lbs   wei8, wei0, 8
  lbs   wei7, wei0, 7
  lbs   wei6, wei0, 6
  lbs   wei5, wei0, 5
  lbs   wei4, wei0, 4
  lbs   wei3, wei0, 3
  lbs   wei2, wei0, 2
  lbs   wei1, wei0, 1
  lbs   wei0, wei0, 0

for_loop:
  //每次进2个input到in23 ----不知道符号(u/s)有没有影响
  lhs   in23,  in,   0
  //前一行
  sub   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in01,  wei6
  add   out0, out0, tmp  
  mul_sh_sl   tmp,  in01,  wei7
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in23,  wei8
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //第二个点
  lbs   out0, out,  1
  mul_sh_sl   tmp,  in01,  wei6
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in23,  wei7
  add   out0, out0, tmp 
  mul_sh_sl   tmp,  in23,  wei8
  add   out0, out0, tmp 
  sb    out,  1,  out0

  //本行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in01,  wei3
  add   out0, out0, tmp  
  mul_sh_sl   tmp,  in01,  wei4
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in23,  wei5
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //第二点
  lbs   out0, out,  1
  mul_sh_sl   tmp,  in01,  wei3
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in23,  wei4
  add   out0, out0, tmp 
  mul_sh_sl   tmp,  in23,  wei5
  add   out0, out0, tmp 
  sb    out,  1,  out0

  //后一行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in01,  wei0
  add   out0, out0, tmp  
  mul_sh_sl   tmp,  in01,  wei1
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in23,  wei2
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //第二点
  lbs   out0, out,  1
  mul_sh_sl   tmp,  in01,  wei0
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in23,  wei1
  add   out0, out0, tmp 
  mul_sh_sl   tmp,  in23,  wei2
  add   out0, out0, tmp 
  sb    out,  1,  out0

  sub   out,  out,  size
  //复用input
  move  in01,  in23
  //load store 不能直接加idx寄存器，还是得自己动
  add   out,  out,  2
  add   in,   in,   2
  //循环条件
  add   idx,  idx,  2
  add   tmp,  size, -2
  jneq  tmp,  idx,  for_loop
  //出循环，out还剩最后2个点（3行）


  //返回
  jump r23



