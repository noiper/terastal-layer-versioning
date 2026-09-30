Network sparse_to_dense_resnet18_upproj {

FPS: 30

Layer conv1 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 64, C: 4, Y: 228, X: 304, R: 7, S: 7 }
}

Layer layer1_0_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 57, X: 76, R: 3, S: 3 }
}

Layer layer1_0_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 57, X: 76, R: 3, S: 3 }
}

Layer layer1_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 64, Y: 57, X: 76, R: 1, S: 1 }
}

Layer layer1_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 57, X: 76, R: 3, S: 3 }
}

Layer layer1_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 57, X: 76, R: 3, S: 3 }
}

Layer layer1_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 64, Y: 57, X: 76, R: 1, S: 1 }
}

Layer layer2_0_conv1 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 128, C: 64, Y: 57, X: 76, R: 3, S: 3 }
}

Layer layer2_0_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 29, X: 38, R: 3, S: 3 }
}

Layer layer2_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 128, Y: 29, X: 38, R: 1, S: 1 }
}

Layer layer2_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 29, X: 38, R: 3, S: 3 }
}

Layer layer2_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 29, X: 38, R: 3, S: 3 }
}

Layer layer2_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 128, Y: 29, X: 38, R: 1, S: 1 }
}

Layer layer3_0_conv1 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 256, C: 128, Y: 29, X: 38, R: 3, S: 3 }
}

Layer layer3_0_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 15, X: 19, R: 3, S: 3 }
}

Layer layer3_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 15, X: 19, R: 1, S: 1 }
}

Layer layer3_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 15, X: 19, R: 3, S: 3 }
}

Layer layer3_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 15, X: 19, R: 3, S: 3 }
}

Layer layer3_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 15, X: 19, R: 1, S: 1 }
}

Layer layer4_0_conv1 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 512, C: 256, Y: 15, X: 19, R: 3, S: 3 }
}

Layer layer4_0_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 8, X: 10, R: 3, S: 3 }
}

Layer layer4_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 8, X: 10, R: 1, S: 1 }
}

Layer layer4_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 8, X: 10, R: 3, S: 3 }
}

Layer layer4_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 8, X: 10, R: 3, S: 3 }
}

Layer layer4_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 8, X: 10, R: 1, S: 1 }
}

Layer conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 512, Y: 8, X: 10, R: 1, S: 1 }
}

Layer decoder_layer1_upper_branch_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 256, Y: 16, X: 20, R: 5, S: 5 }
}

Layer decoder_layer1_upper_branch_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 16, X: 20, R: 3, S: 3 }
}

Layer decoder_layer1_bottom_branch_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 256, Y: 16, X: 20, R: 5, S: 5 }
}

Layer decoder_layer2_upper_branch_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 128, Y: 32, X: 40, R: 5, S: 5 }
}

Layer decoder_layer2_upper_branch_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 32, X: 40, R: 3, S: 3 }
}

Layer decoder_layer2_bottom_branch_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 128, Y: 32, X: 40, R: 5, S: 5 }
}

Layer decoder_layer3_upper_branch_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 64, Y: 64, X: 80, R: 5, S: 5 }
}

Layer decoder_layer3_upper_branch_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 32, Y: 64, X: 80, R: 3, S: 3 }
}

Layer decoder_layer3_bottom_branch_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 64, Y: 64, X: 80, R: 5, S: 5 }
}

Layer decoder_layer4_upper_branch_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 16, C: 32, Y: 128, X: 160, R: 5, S: 5 }
}

Layer decoder_layer4_upper_branch_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 16, C: 16, Y: 128, X: 160, R: 3, S: 3 }
}

Layer decoder_layer4_bottom_branch_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 16, C: 32, Y: 128, X: 160, R: 5, S: 5 }
}

Layer conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 16, Y: 128, X: 160, R: 3, S: 3 }
}

}
