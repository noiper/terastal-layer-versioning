Network mb2-ssd-lite_3 {

FPS: 30

Layer base_net_0_0 { 
	Type: CONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 32, C: 3, Y: 300, X: 300, R: 3, S: 3 }
}

Layer base_net_1_conv_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 32, Y: 150, X: 150, R: 3, S: 3 }
}

Layer base_net_1_conv_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 16, C: 32, Y: 150, X: 150, R: 1, S: 1 }
}

Layer base_net_2_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 16, Y: 150, X: 150, R: 1, S: 1 }
}

Layer base_net_2_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 96, Y: 150, X: 150, R: 3, S: 3 }
}

Layer base_net_2_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 96, Y: 75, X: 75, R: 1, S: 1 }
}

Layer base_net_3_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 144, C: 24, Y: 75, X: 75, R: 1, S: 1 }
}

Layer base_net_3_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 144, Y: 75, X: 75, R: 3, S: 3 }
}

Layer base_net_3_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 144, Y: 75, X: 75, R: 1, S: 1 }
}

Layer base_net_4_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 144, C: 24, Y: 75, X: 75, R: 1, S: 1 }
}

Layer base_net_4_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 144, Y: 75, X: 75, R: 3, S: 3 }
}

Layer base_net_4_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 144, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_5_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 32, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_5_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 192, Y: 38, X: 38, R: 3, S: 3 }
}

Layer base_net_5_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 192, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_6_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 32, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_6_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 192, Y: 38, X: 38, R: 3, S: 3 }
}

Layer base_net_6_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 32, C: 192, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_7_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 32, Y: 38, X: 38, R: 1, S: 1 }
}

Layer base_net_7_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 192, Y: 38, X: 38, R: 3, S: 3 }
}

Layer base_net_7_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 192, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_8_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 64, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_8_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 384, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_8_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 384, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_9_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 64, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_9_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 384, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_9_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 384, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_10_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 64, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_10_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 384, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_10_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 384, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_11_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 64, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_11_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 384, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_11_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 384, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_12_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 576, C: 96, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_12_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 576, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_12_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 576, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_13_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 576, C: 96, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_13_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 576, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_13_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 576, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_14_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 576, C: 96, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_14_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 576, Y: 19, X: 19, R: 3, S: 3 }
}

Layer base_net_14_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 576, Y: 10, X: 10, R: 1, S: 1 }
}

Layer classification_headers_0_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 576, Y: 19, X: 19, R: 3, S: 3 }
}

Layer classification_headers_0_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 576, Y: 19, X: 19, R: 1, S: 1 }
}

Layer regression_headers_0_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 576, Y: 19, X: 19, R: 3, S: 3 }
}

Layer regression_headers_0_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 576, Y: 19, X: 19, R: 1, S: 1 }
}

Layer base_net_15_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 960, C: 160, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_15_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 960, Y: 10, X: 10, R: 3, S: 3 }
}

Layer base_net_15_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 960, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_16_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 960, C: 160, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_16_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 960, Y: 10, X: 10, R: 3, S: 3 }
}

Layer base_net_16_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 160, C: 960, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_17_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 960, C: 160, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_17_conv_3 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 960, Y: 10, X: 10, R: 3, S: 3 }
}

Layer base_net_17_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 320, C: 960, Y: 10, X: 10, R: 1, S: 1 }
}

Layer base_net_18_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1280, C: 320, Y: 10, X: 10, R: 1, S: 1 }
}

Layer classification_headers_1_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1280, Y: 10, X: 10, R: 3, S: 3 }
}

Layer classification_headers_1_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 1280, Y: 10, X: 10, R: 1, S: 1 }
}

Layer regression_headers_1_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 1280, Y: 10, X: 10, R: 3, S: 3 }
}

Layer regression_headers_1_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 1280, Y: 10, X: 10, R: 1, S: 1 }
}

Layer extras_0_conv_0_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 320, Y: 20, X: 20, R: 1, S: 1 }
}

Layer extras_0_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 10, X: 10, R: 3, S: 3 }
}

Layer extras_0_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 256, Y: 5, X: 5, R: 1, S: 1 }
}

Layer classification_headers_2_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 5, X: 5, R: 3, S: 3 }
}

Layer classification_headers_2_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 512, Y: 5, X: 5, R: 1, S: 1 }
}

Layer regression_headers_2_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 512, Y: 5, X: 5, R: 3, S: 3 }
}

Layer regression_headers_2_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 512, Y: 5, X: 5, R: 1, S: 1 }
}

Layer extras_1_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 512, Y: 5, X: 5, R: 1, S: 1 }
}

Layer extras_1_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 128, Y: 5, X: 5, R: 3, S: 3 }
}

Layer extras_1_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 128, Y: 3, X: 3, R: 1, S: 1 }
}

Layer classification_headers_3_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 3, X: 3, R: 3, S: 3 }
}

Layer classification_headers_3_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 256, Y: 3, X: 3, R: 1, S: 1 }
}

Layer regression_headers_3_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 3, X: 3, R: 3, S: 3 }
}

Layer regression_headers_3_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 256, Y: 3, X: 3, R: 1, S: 1 }
}

Layer extras_2_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 256, Y: 3, X: 3, R: 1, S: 1 }
}

Layer extras_2_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 128, Y: 3, X: 3, R: 3, S: 3 }
}

Layer extras_2_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 128, Y: 2, X: 2, R: 1, S: 1 }
}

Layer classification_headers_4_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 2, X: 2, R: 3, S: 3 }
}

Layer classification_headers_4_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 256, Y: 2, X: 2, R: 1, S: 1 }
}

Layer regression_headers_4_0 { 
	Type: DSCONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1, C: 256, Y: 2, X: 2, R: 3, S: 3 }
}

Layer regression_headers_4_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 256, Y: 2, X: 2, R: 1, S: 1 }
}

Layer extras_3_conv_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 256, Y: 2, X: 2, R: 1, S: 1 }
}

Layer extras_3_conv_3 { 
	Type: DSCONV
	Stride { X: 2, Y: 2 }
	Dimensions: { N: 1, K: 1, C: 64, Y: 2, X: 2, R: 3, S: 3 }
}

Layer extras_3_conv_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 64, Y: 1, X: 1, R: 1, S: 1 }
}

Layer classification_headers_5 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 126, C: 64, Y: 1, X: 1, R: 1, S: 1 }
}

Layer regression_headers_5 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 24, C: 64, Y: 1, X: 1, R: 1, S: 1 }
}

}
