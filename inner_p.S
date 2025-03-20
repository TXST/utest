
.globl inner_p
inner_p:
#define xx r0
#define out0 r1
#define size r2

#define B0 r3
#define B1 r4
#define B2 r5
#define B3 r6

#define A0 r7
#define A1 r8
#define A2 r9
#define A3 r10

#define out r11

#define idx r13
#define tmp r15

//void inner_p(T* xx,T* oo,int16_t size,T* BB)
//x 右移，每次尽量多娶几个
//B 4个数固定在寄存器

  lbs   B3, B0, 3
  lbs   B2, B0, 2
  lbs   B1, B0, 1
  lbs   B0, B0, 0

move  idx,  1

for_loop:
  //每次进4个x
  lbs   A3, xx, 672       //避免零碎加载，组织成（W, cin, H）
  lbs   A2, xx, 448
  lbs   A1, xx, 224
  lbs   A0, xx, 0

  lbs   out, out0,  0           
  mul_sl_sl   tmp,  A0,  B0
  add   out, out, tmp  

  mul_sl_sl   tmp,  A1,  B1
  add   out, out, tmp 

  mul_sl_sl   tmp,  A2,  B2
  add   out, out, tmp 

  mul_sl_sl   tmp,  A3,  B3
  add   out, out, tmp  
  sb    out0,  0,  out

  //load store 不能直接加idx寄存器，还是得自己动
  add   out0,  out0,  1           
  add   xx,   xx,   1
  //循环条件
  add   idx,  idx,  1
  jneq  size,  idx,  for_loop

  //返回
  jump r23




