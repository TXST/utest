
.globl inner_p
inner_p:
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

#define A0 r11
#define A1 r12
#define A2 r13
#define A3 r14
#define A4 r15
#define A5 r16
#define A6 r17
#define A7 r18

#define out r19

#define idx r20
#define tmp r21

//void inner_p(T* xx,T* oo,int16_t size,T* BB)
//x 右移，每次尽量多娶几个
//B 16个数固定在寄存器
  lhs   B7, B0, 14
  lhs   B6, B0, 12
  lhs   B5, B0, 10
  lhs   B4, B0, 8
  lhs   B3, B0, 6
  lhs   B2, B0, 4
  lhs   B1, B0, 2
  lhs   B0, B0, 0

move  idx,  1

for_loop:
  //每次进16x
  //cin连续才能一次加载多个，so组织成（cin,W,H），加载碎了，先不管，估计要重排
  lhs   A7, xx, 14       
  lhs   A6, xx, 12
  lhs   A5, xx, 10
  lhs   A4, xx, 8
  lhs   A3, xx, 6       
  lhs   A2, xx, 4
  lhs   A1, xx, 2
  lhs   A0, xx, 0

  lbs   out, out0,  0  
  mul_sl_sl   tmp,  A0,  B0
  add   out, out, tmp  
  mul_sh_sh   tmp,  A0,  B0
  add   out, out, tmp  

  mul_sl_sl   tmp,  A1,  B1
  add   out, out, tmp 
  mul_sh_sh   tmp,  A1,  B1
  add   out, out, tmp 

  mul_sl_sl   tmp,  A2,  B2
  add   out, out, tmp
  mul_sh_sh   tmp,  A2,  B2
  add   out, out, tmp 

  mul_sl_sl   tmp,  A3,  B3
  add   out, out, tmp  
  mul_sh_sh   tmp,  A3,  B3
  add   out, out, tmp  

  mul_sl_sl   tmp,  A4,  B4
  add   out, out, tmp  
  mul_sh_sh   tmp,  A4,  B4
  add   out, out, tmp 

  mul_sl_sl   tmp,  A5,  B5
  add   out, out, tmp  
  mul_sh_sh   tmp,  A5,  B5
  add   out, out, tmp 

  mul_sl_sl   tmp,  A6,  B6
  add   out, out, tmp  
  mul_sh_sh   tmp,  A6,  B6
  add   out, out, tmp 

  mul_sl_sl   tmp,  A7,  B7
  add   out, out, tmp  
  mul_sh_sh   tmp,  A7,  B7
  add   out, out, tmp 

  sb    out0,  0,  out

  //load store 不能直接加idx寄存器，还是得自己动
  add   out0,  out0,  1           
  add   xx,   xx,   16
  //循环条件
  add   idx,  idx,  1
  jneq  size,  idx,  for_loop

  //返回
  jump r23




