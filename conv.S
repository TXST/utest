
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

#define in0 r12
#define in1 r13
#define in2 r14
#define out0 r15

#define idx r16
#define tmp r17

//void conv_1d(T* in,T* out,int16_t size,T* wei)
//in 右移，每次尽量多娶几个
//wei 9个数固定在寄存器，先用完一个通道再换另一个通道
//第一行、最后一行、第一个点、最后一个点


  move  idx,  0
  //in
  move  in0,  0
  lbs   in1,  in,   0
  add   in,   in,   1
  //weight
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
  //每次进一个input到in2
  lbs   in2,  in,   0
  //前一行
  sub   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei6
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei7
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in2,  wei8
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //本行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei3
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei4
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in2,  wei5
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //后一行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei0
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei1
  add   out0, out0, tmp 
  mul_sl_sl   tmp,  in2,  wei2
  add   out0, out0, tmp 
  sb    out,  0,  out0
  sub   out,  out,  size
  //复用input
  move  in0,  in1
  move  in1,  in2
  //load store 不能直接加idx寄存器，还是得自己动
  add   out,  out,  1
  add   in,   in,   1
  //循环条件
  add   idx,  idx,  1
  add   tmp,  size, -1
  jneq  tmp,  idx,  for_loop
  //出循环，out还剩最后一个点（3行）
  //前一行
  sub   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei6
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei7
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //本行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei3
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei4
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //后一行
  add   out,  out,  size
  lbs   out0, out,  0
  mul_sl_sl   tmp,  in0,  wei0
  add   out0, out0, tmp  
  mul_sl_sl   tmp,  in1,  wei1
  add   out0, out0, tmp 
  sb    out,  0,  out0
  //返回
  jump r23




