Network inceptionv3 {

FPS: 30

Layer Conv2d_1a_3x3_conv { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 32, C: 3, Y: 299, X: 299, R: 3, S: 3 }
}

Layer Conv2d_2a_3x3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 32, Y: 149, X: 149, R: 3, S: 3 }
}

Layer Conv2d_2b_3x3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 32, Y: 147, X: 147, R: 3, S: 3 }
}

Layer Conv2d_3b_1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 80, C: 64, Y: 73, X: 73, R: 1, S: 1 }
}

Layer Conv2d_4a_3x3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 80, Y: 73, X: 73, R: 3, S: 3 }
}

Layer Mixed_5b_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 192, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5b_branch5x5_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 48, C: 192, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5b_branch5x5_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 48, Y: 35, X: 35, R: 5, S: 5 }
}

Layer Mixed_5b_branch3x3dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 192, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5b_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 64, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5b_branch3x3dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 96, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5b_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 192, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5c_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5c_branch5x5_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 48, C: 256, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5c_branch5x5_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 48, Y: 35, X: 35, R: 5, S: 5 }
}

Layer Mixed_5c_branch3x3dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5c_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 64, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5c_branch3x3dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 96, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5c_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5d_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 288, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5d_branch5x5_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 48, C: 288, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5d_branch5x5_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 48, Y: 35, X: 35, R: 5, S: 5 }
}

Layer Mixed_5d_branch3x3dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 288, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_5d_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 64, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5d_branch3x3dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 96, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_5d_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 288, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_6a_branch3x3_conv { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 384, C: 288, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_6a_branch3x3dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 288, Y: 35, X: 35, R: 1, S: 1 }
}

Layer Mixed_6a_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 64, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_6a_branch3x3dbl_3_conv { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 96, C: 96, Y: 35, X: 35, R: 3, S: 3 }
}

Layer Mixed_6b_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6b_branch7x7_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6b_branch7x7_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6b_branch7x7_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 128, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6b_branch7x7dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6b_branch7x7dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6b_branch7x7dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6b_branch7x7dbl_4_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6b_branch7x7dbl_5_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 128, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6b_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6c_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6c_branch7x7_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6c_branch7x7_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6c_branch7x7_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6c_branch7x7dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6c_branch7x7dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6c_branch7x7dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6c_branch7x7dbl_4_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6c_branch7x7dbl_5_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6c_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6d_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6d_branch7x7_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6d_branch7x7_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6d_branch7x7_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6d_branch7x7dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6d_branch7x7dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6d_branch7x7dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6d_branch7x7dbl_4_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 160, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6d_branch7x7dbl_5_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 160, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6d_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6e_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6e_branch7x7_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6e_branch7x7_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6e_branch7x7_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6e_branch7x7dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_6e_branch7x7dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6e_branch7x7dbl_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6e_branch7x7dbl_4_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_6e_branch7x7dbl_5_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_6e_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_7a_branch3x3_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_7a_branch3x3_2_conv { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 320, C: 192, Y: 17, X: 17, R: 3, S: 3 }
}

Layer Mixed_7a_branch7x7x3_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 17, X: 17, R: 1, S: 1 }
}

Layer Mixed_7a_branch7x7x3_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 1, S: 7 }
}

Layer Mixed_7a_branch7x7x3_3_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 7, S: 1 }
}

Layer Mixed_7a_branch7x7x3_4_conv { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 192, C: 192, Y: 17, X: 17, R: 3, S: 3 }
}

Layer Mixed_7b_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 320, C: 1280, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7b_branch3x3_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1280, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7b_branch3x3_2a_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 1, S: 3 }
}

Layer Mixed_7b_branch3x3_2b_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 3, S: 1 }
}

Layer Mixed_7b_branch3x3dbl_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 448, C: 1280, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7b_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 448, Y: 8, X: 8, R: 3, S: 3 }
}

Layer Mixed_7b_branch3x3dbl_3a_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 1, S: 3 }
}

Layer Mixed_7b_branch3x3dbl_3b_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 3, S: 1 }
}

Layer Mixed_7b_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 1280, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7c_branch1x1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 320, C: 2048, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7c_branch3x3_1_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 2048, Y: 8, X: 8, R: 1, S: 1 }
}

Layer Mixed_7c_branch3x3_2a_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 1, S: 3 }
}

Layer Mixed_7c_branch3x3_2b_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 3, S: 1 }
}

Layer Mixed_7c_branch3x3dbl_1_conv_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 112, C: 512, Y: 16, X: 16, R: 1, S: 1 }
}

Layer Mixed_7c_branch3x3dbl_2_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 448, Y: 8, X: 8, R: 3, S: 3 }
}

Layer Mixed_7c_branch3x3dbl_3a_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 1, S: 3 }
}

Layer Mixed_7c_branch3x3dbl_3b_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 384, Y: 8, X: 8, R: 3, S: 1 }
}

Layer Mixed_7c_branch_pool_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 2048, Y: 8, X: 8, R: 1, S: 1 }
}

Layer fc { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1000, C: 2048, Y: 1, X: 1, R: 1, S: 1 }
}

}
