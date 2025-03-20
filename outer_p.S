
.globl outer_p
outer_p:
#define xx r0
#define out0 r1
#define size r2

#define B0 r3
#define B1 r4
#define B2 r5
#define B3 r6
#define B4 r7
#define B5 r8
#define B6 r9
#define B7 r10

#define out r11
#define xi r12

#define idx r13
#define tmp r15

//void outer_p(T* xx,T* oo,int16_t size,T* BB)
//x 右移，每次尽量多娶几个
//B 8个数固定在寄存器

  //B
  lbs   B7, B0, 7
  lbs   B6, B0, 6
  lbs   B5, B0, 5
  lbs   B4, B0, 4
  lbs   B3, B0, 3
  lbs   B2, B0, 2
  lbs   B1, B0, 1
  lbs   B0, B0, 0

move  idx,  1

for_loop:
  //每次进一个x
  lbs   xi,  xx,   0

  lbs   out, out0,  0           //这里假设out0列存（=B=128）
  mul_sl_sl   tmp,  xi,  B0
  add   out, out, tmp  
  sb    out0,  0,  out

  lbs   out, out0,  1
  mul_sl_sl   tmp,  xi,  B1
  add   out, out, tmp 
  sb    out0,  1,  out

  lbs   out, out0,  2
  mul_sl_sl   tmp,  xi,  B2
  add   out, out, tmp 
  sb    out0,  2,  out

  lbs   out, out0,  3
  mul_sl_sl   tmp,  xi,  B3
  add   out, out, tmp  
  sb    out0,  3,  out

  lbs   out, out0,  4
  mul_sl_sl   tmp,  xi,  B4
  add   out, out, tmp 
  sb    out0,  4,  out

  lbs   out, out0,  5
  mul_sl_sl   tmp,  xi,  B5
  add   out, out, tmp 
  sb    out0,  5,  out
  
  lbs   out, out0,  6
  mul_sl_sl   tmp,  xi,  B6
  add   out, out, tmp 
  sb    out0,  6,  out

  lbs   out, out0,  7
  mul_sl_sl   tmp,  xi,  B7
  add   out, out, tmp 
  sb    out0,  7,  out

  //load store 不能直接加idx寄存器，还是得自己动
  add   out0,  out0,  128           //列存（B=128）
  add   xx,   xx,   1
  //循环条件
  add   idx,  idx,  1
  jneq  size,  idx,  for_loop

  //返回
  jump r23




