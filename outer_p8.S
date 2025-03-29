
.globl outer_p
outer_p:
#define x_addr r0
#define out_addr r1
#define size r2
#define B_addr r3

//d寄存器只有偶数；此时较小的r寄存器装高字节
#define BB d4 //(r4+r5)
#define BB0 r5
#define BB1 r4
#define xx d6 //(r6+r7)
#define xx0 r7
#define xx1 r6
#define out d8  //(r8+r9)
#define oo0 r9
#define oo1 r8

#define xtmp r10
#define btmp r11

#define idx r13
#define tmp r15

//void outer_p(T* xx,T* oo,int16_t size,T* BB)
//B 8个数固定在寄存器

  ld  BB, B_addr, 0

move  idx,  1

for_loop:
  //每次进8x
  ld   xx,  x_addr,   0
  ld   out, out_addr,  0       //如果之后的GEMV用向量内积，则out列存；如果用查表则行存，这里先随便选一个列存

//-------------B * x0--------------------

  mul_sl_sl   tmp,  xx0,  BB0     //x0b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sl_sh   tmp,  xx0,  BB0     //x0b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16           //BB0右移，b2b3入场
  
  mul_sl_sl   tmp,  xx0,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sl_sh   tmp,  xx0,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sl_sl   tmp,  xx0,  BB1     //x0b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sl_sh   tmp,  xx0,  BB1     //x0b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sl_sl   tmp,  xx0,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sl_sh   tmp,  xx0,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

//-------------B * x1--------------------

  mul_sh_sl   tmp,  xx0,  BB0     //x1b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sh_sh   tmp,  xx0,  BB0     //x1b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16
  
  mul_sh_sl   tmp,  xx0,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sh_sh   tmp,  xx0,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sh_sl   tmp,  xx0,  BB1     //x1b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sh_sh   tmp,  xx0,  BB1     //x1b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sh_sl   tmp,  xx0,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sh_sh   tmp,  xx0,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

  lsr   xtmp,  xx0,  16               //xx0右移，x2x3入场

//------------B * x2------------------

  mul_sl_sl   tmp,  xtmp,  BB0     //x2b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sl_sh   tmp,  xtmp,  BB0     //x2b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16           //BB0右移，b2b3入场
  
  mul_sl_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sl_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sl_sl   tmp,  xtmp,  BB1     //x2b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sl_sh   tmp,  xtmp,  BB1     //x2b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sl_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sl_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

//-------------B * x3--------------------

  mul_sh_sl   tmp,  xtmp,  BB0     //x3b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sh_sh   tmp,  xtmp,  BB0     //x3b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16
  
  mul_sh_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sh_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sh_sl   tmp,  xtmp,  BB1     //x3b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sh_sh   tmp,  xtmp,  BB1     //x3b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sh_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sh_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

//-------------B * x4--------------------

  mul_sl_sl   tmp,  xx1,  BB0     //x4b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sl_sh   tmp,  xx1,  BB0     //x4b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16
  
  mul_sl_sl   tmp,  xx1,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sl_sh   tmp,  xx1,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sl_sl   tmp,  xx1,  BB1     //x4b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sl_sh   tmp,  xx1,  BB1     //x4b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sl_sl   tmp,  xx1,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sl_sh   tmp,  xx1,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

//-------------B * x5--------------------

  mul_sh_sl   tmp,  xx1,  BB0     //x5b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sh_sh   tmp,  xx1,  BB0     //x5b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16
  
  mul_sh_sl   tmp,  xx1,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sh_sh   tmp,  xx1,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sh_sl   tmp,  xx1,  BB1     //x5b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sh_sh   tmp,  xx1,  BB1     //x5b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sh_sl   tmp,  xx1,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sh_sh   tmp,  xx1,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

  lsr   xtmp,  xx1,  16               //x6x7入场

  //------------B * x6------------------

    mul_sl_sl   tmp,  xtmp,  BB0     //x6b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sl_sh   tmp,  xtmp,  BB0     //x6b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16           //BB0右移，b2b3入场
  
  mul_sl_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sl_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sl_sl   tmp,  xtmp,  BB1     //x6b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sl_sh   tmp,  xtmp,  BB1     //x6b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sl_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sl_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列

//-------------B * x7--------------------

  mul_sh_sl   tmp,  xtmp,  BB0     //x7b0
  add   tmp,  tmp,  oo0
  sb    out_addr,  0,  tmp
  mul_sh_sh   tmp,  xtmp,  BB0     //x7b1
  lsr_add   tmp,  tmp,  oo0,  8
  sb    out_addr,  1,  tmp 
  
  lsr   btmp,  BB0,  16
  
  mul_sh_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo0,  16    
  sb    out_addr,  2,  tmp
  
  mul_sh_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo0,  24  
  sb    out_addr,  3,  tmp 

  mul_sh_sl   tmp,  xtmp,  BB1     //x7b4
  lsr_add   tmp,  tmp,  oo1,  0
  sb    out_addr,  4,  tmp
  mul_sh_sh   tmp,  xtmp,  BB1     //x7b5
  lsr_add   tmp,  tmp,  oo1,  8
  sb    out_addr,  5,  tmp 

  lsr   btmp,  BB1,  16
  
  mul_sh_sl   tmp,  xtmp,  btmp 
  lsr_add   tmp,  tmp,  oo1,  16    
  sb    out_addr,  6,  tmp
  
  mul_sh_sh   tmp,  xtmp,  btmp   
  lsr_add   tmp,  tmp,  oo1,  24  
  sb    out_addr,  7,  tmp 

  add   out_addr,   out_addr,   128   //output右移一列


  //load store 不能直接加idx寄存器，还是得自己动
  add   x_addr,   x_addr,   8         //每次进8个
  //循环条件
  add   idx,  idx,  1
  jneq  size,  idx,  for_loop

  //返回
  jump r23