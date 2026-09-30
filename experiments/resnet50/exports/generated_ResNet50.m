Network ResNet50 {

FPS: 30

Layer conv1 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 64, C: 3, Y: 224, X: 224, R: 7, S: 7 }
}

Layer layer1_0_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_0_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 56, X: 56, R: 3, S: 3 }
}

Layer layer1_0_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 64, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 56, X: 56, R: 3, S: 3 }
}

Layer layer1_1_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 64, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_2_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_2_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 56, X: 56, R: 3, S: 3 }
}

Layer layer1_2_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 64, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer1_2_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer2_0_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 256, Y: 56, X: 56, R: 1, S: 1 }
}

Layer layer2_0_conv2 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 56, X: 56, R: 3, S: 3 }
}

Layer layer2_0_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 128, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 28, X: 28, R: 3, S: 3 }
}

Layer layer2_1_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 128, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_2_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_2_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 28, X: 28, R: 3, S: 3 }
}

Layer layer2_2_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 128, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_2_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_3_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_3_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 28, X: 28, R: 3, S: 3 }
}

Layer layer2_3_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 128, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer2_3_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer3_0_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 512, Y: 28, X: 28, R: 1, S: 1 }
}

Layer layer3_0_conv2 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 28, X: 28, R: 3, S: 3 }
}

Layer layer3_0_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer3_1_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_2_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_2_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer3_2_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_2_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_3_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_3_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer3_3_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_3_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_4_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_4_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer3_4_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_4_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_5_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_5_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer3_5_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1024, C: 256, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer3_5_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer4_0_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 1024, Y: 14, X: 14, R: 1, S: 1 }
}

Layer layer4_0_conv2 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 14, X: 14, R: 3, S: 3 }
}

Layer layer4_0_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 2048, C: 512, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_0_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 2048, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_1_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 2048, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_1_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 7, X: 7, R: 3, S: 3 }
}

Layer layer4_1_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 2048, C: 512, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_1_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 2048, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_2_conv1 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 2048, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_2_conv2 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 7, X: 7, R: 3, S: 3 }
}

Layer layer4_2_conv3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 2048, C: 512, Y: 7, X: 7, R: 1, S: 1 }
}

Layer layer4_2_Residual { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 2048, Y: 7, X: 7, R: 1, S: 1 }
}

Layer fc { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1000, C: 2048, Y: 1, X: 1, R: 1, S: 1 }
}

}
