#if 0
//
// Generated by Microsoft (R) HLSL Shader Compiler 10.1
//
//
// Buffer Definitions: 
//
// cbuffer ConstantBufferCS
// {
//
//   uint num;                          // Offset:    0 Size:     4
//   uint channel;                      // Offset:    4 Size:     4
//   uint size;                         // Offset:    8 Size:     4
//
// }
//
//
// Resource Bindings:
//
// Name                                 Type  Format         Dim      ID      HLSL Bind  Count
// ------------------------------ ---------- ------- ----------- ------- -------------- ------
// input_buffer                          UAV   float         buf      U0             u0      1 
// output_buffer                         UAV   float         buf      U1             u1      1 
// ConstantBufferCS                  cbuffer      NA          NA     CB0            cb0      1 
//
//
//
// Input signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Input
//
// Output signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Output
cs_5_1
dcl_globalFlags refactoringAllowed
dcl_constantbuffer CB0[0:0][1], immediateIndexed, space=0
dcl_uav_typed_buffer (float,float,float,float) U0[0:0], space=0
dcl_uav_typed_buffer (float,float,float,float) U1[1:1], space=0
dcl_input vThreadGroupID.x
dcl_input vThreadIDInGroup.x
dcl_temps 2
dcl_thread_group 512, 1, 1
imad r0.x, vThreadGroupID.x, l(512), vThreadIDInGroup.x
imul null, r0.y, CB0[0][0].z, CB0[0][0].y
imul null, r0.z, r0.y, CB0[0][0].x
ult r0.z, r0.x, r0.z
if_nz r0.z
  udiv r0.z, null, r0.x, r0.y
  imul null, r0.w, r0.y, r0.z
  uge r1.x, r0.x, r0.y
  iadd r0.w, -r0.w, r0.x
  movc r0.w, r1.x, r0.w, r0.x
  udiv r0.w, null, r0.w, CB0[0][0].y
  udiv null, r1.x, r0.x, CB0[0][0].y
  imad r0.w, r1.x, CB0[0][0].z, r0.w
  imad r0.y, r0.z, r0.y, r0.w
  ld_uav_typed r0.y, r0.yyyy, U0[0].yxzw
  store_uav_typed U1[1].xyzw, r0.xxxx, r0.yyyy
endif 
ret 
// Approximately 18 instruction slots used
#endif

const BYTE g_format_half_output[] =
{
     68,  88,  66,  67,  64, 244, 
    223,  27, 201, 159, 113, 143, 
    167, 191,  82, 207, 152,   6, 
    184, 111,   1,   0,   0,   0, 
    132,   5,   0,   0,   5,   0, 
      0,   0,  52,   0,   0,   0, 
     20,   2,   0,   0,  36,   2, 
      0,   0,  52,   2,   0,   0, 
    232,   4,   0,   0,  82,  68, 
     69,  70, 216,   1,   0,   0, 
      1,   0,   0,   0, 224,   0, 
      0,   0,   3,   0,   0,   0, 
     60,   0,   0,   0,   1,   5, 
     83,  67,   0,   5,   0,   0, 
    173,   1,   0,   0,  19,  19, 
     68,  37,  60,   0,   0,   0, 
     24,   0,   0,   0,  40,   0, 
      0,   0,  40,   0,   0,   0, 
     36,   0,   0,   0,  12,   0, 
      0,   0,   0,   0,   0,   0, 
    180,   0,   0,   0,   4,   0, 
      0,   0,   5,   0,   0,   0, 
      1,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 193,   0, 
      0,   0,   4,   0,   0,   0, 
      5,   0,   0,   0,   1,   0, 
      0,   0, 255, 255, 255, 255, 
      1,   0,   0,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0, 207,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    105, 110, 112, 117, 116,  95, 
     98, 117, 102, 102, 101, 114, 
      0, 111, 117, 116, 112, 117, 
    116,  95,  98, 117, 102, 102, 
    101, 114,   0,  67, 111, 110, 
    115, 116,  97, 110, 116,  66, 
    117, 102, 102, 101, 114,  67, 
     83,   0, 207,   0,   0,   0, 
      3,   0,   0,   0, 248,   0, 
      0,   0,  16,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 112,   1,   0,   0, 
      0,   0,   0,   0,   4,   0, 
      0,   0,   2,   0,   0,   0, 
    124,   1,   0,   0,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
    160,   1,   0,   0,   4,   0, 
      0,   0,   4,   0,   0,   0, 
      2,   0,   0,   0, 124,   1, 
      0,   0,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0, 168,   1, 
      0,   0,   8,   0,   0,   0, 
      4,   0,   0,   0,   2,   0, 
      0,   0, 124,   1,   0,   0, 
      0,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 110, 117, 109,   0, 
    100, 119, 111, 114, 100,   0, 
    171, 171,   0,   0,  19,   0, 
      1,   0,   1,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 116,   1, 
      0,   0,  99, 104,  97, 110, 
    110, 101, 108,   0, 115, 105, 
    122, 101,   0,  77, 105,  99, 
    114, 111, 115, 111, 102, 116, 
     32,  40,  82,  41,  32,  72, 
     76,  83,  76,  32,  83, 104, 
     97, 100, 101, 114,  32,  67, 
    111, 109, 112, 105, 108, 101, 
    114,  32,  49,  48,  46,  49, 
      0, 171, 171, 171,  73,  83, 
     71,  78,   8,   0,   0,   0, 
      0,   0,   0,   0,   8,   0, 
      0,   0,  79,  83,  71,  78, 
      8,   0,   0,   0,   0,   0, 
      0,   0,   8,   0,   0,   0, 
     83,  72,  69,  88, 172,   2, 
      0,   0,  81,   0,   5,   0, 
    171,   0,   0,   0, 106,   8, 
      0,   1,  89,   0,   0,   7, 
     70, 142,  48,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
    156,   8,   0,   7,  70, 238, 
     49,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,  85,  85,   0,   0, 
      0,   0,   0,   0, 156,   8, 
      0,   7,  70, 238,  49,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0,   1,   0,   0,   0, 
     85,  85,   0,   0,   0,   0, 
      0,   0,  95,   0,   0,   2, 
     18,  16,   2,   0,  95,   0, 
      0,   2,  18,  32,   2,   0, 
    104,   0,   0,   2,   2,   0, 
      0,   0, 155,   0,   0,   4, 
      0,   2,   0,   0,   1,   0, 
      0,   0,   1,   0,   0,   0, 
     35,   0,   0,   7,  18,   0, 
     16,   0,   0,   0,   0,   0, 
     10,  16,   2,   0,   1,  64, 
      0,   0,   0,   2,   0,   0, 
     10,  32,   2,   0,  38,   0, 
      0,  12,   0, 208,   0,   0, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  42, 128,  48,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
     26, 128,  48,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  38,   0, 
      0,  10,   0, 208,   0,   0, 
     66,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      0,   0,   0,   0,  10, 128, 
     48,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,  79,   0,   0,   7, 
     66,   0,  16,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,  42,   0, 
     16,   0,   0,   0,   0,   0, 
     31,   0,   4,   3,  42,   0, 
     16,   0,   0,   0,   0,   0, 
     78,   0,   0,   8,  66,   0, 
     16,   0,   0,   0,   0,   0, 
      0, 208,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  38,   0,   0,   8, 
      0, 208,   0,   0, 130,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  42,   0,  16,   0, 
      0,   0,   0,   0,  80,   0, 
      0,   7,  18,   0,  16,   0, 
      1,   0,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  30,   0,   0,   8, 
    130,   0,  16,   0,   0,   0, 
      0,   0,  58,   0,  16, 128, 
     65,   0,   0,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,  55,   0, 
      0,   9, 130,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   1,   0,   0,   0, 
     58,   0,  16,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,  78,   0, 
      0,  10, 130,   0,  16,   0, 
      0,   0,   0,   0,   0, 208, 
      0,   0,  58,   0,  16,   0, 
      0,   0,   0,   0,  26, 128, 
     48,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,  78,   0,   0,  10, 
      0, 208,   0,   0,  18,   0, 
     16,   0,   1,   0,   0,   0, 
     10,   0,  16,   0,   0,   0, 
      0,   0,  26, 128,  48,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
     35,   0,   0,  11, 130,   0, 
     16,   0,   0,   0,   0,   0, 
     10,   0,  16,   0,   1,   0, 
      0,   0,  42, 128,  48,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
     58,   0,  16,   0,   0,   0, 
      0,   0,  35,   0,   0,   9, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  42,   0,  16,   0, 
      0,   0,   0,   0,  26,   0, 
     16,   0,   0,   0,   0,   0, 
     58,   0,  16,   0,   0,   0, 
      0,   0, 163,   0,   0,   8, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  86,   5,  16,   0, 
      0,   0,   0,   0,  22, 238, 
     33,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 164,   0, 
      0,   8, 242, 224,  33,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0,   6,   0,  16,   0, 
      0,   0,   0,   0,  86,   5, 
     16,   0,   0,   0,   0,   0, 
     21,   0,   0,   1,  62,   0, 
      0,   1,  83,  84,  65,  84, 
    148,   0,   0,   0,  18,   0, 
      0,   0,   2,   0,   0,   0, 
      0,   0,   0,   0,   2,   0, 
      0,   0,   0,   0,   0,   0, 
      7,   0,   0,   0,   5,   0, 
      0,   0,   1,   0,   0,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0
};
