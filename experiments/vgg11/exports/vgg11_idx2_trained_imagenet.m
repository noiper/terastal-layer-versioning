Network vgg11 {

FPS: 30

Layer features_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 64, C: 3, Y: 224, X: 224, R: 3, S: 3 }
}

Layer features_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 64, Y: 112, X: 112, R: 3, S: 3 }
}

Layer features_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 128, Y: 56, X: 56, R: 3, S: 3 }
}

Layer features_8 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 256, C: 256, Y: 56, X: 56, R: 3, S: 3 }
}

Layer features_11 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 256, Y: 28, X: 28, R: 3, S: 3 }
}

Layer features_13 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 28, X: 28, R: 3, S: 3 }
}

Layer features_16 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 512, C: 512, Y: 14, X: 14, R: 3, S: 3 }
}

Layer features_18_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 128, C: 128, Y: 28, X: 28, R: 3, S: 3 }
}

Layer classifier_0 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 4096, C: 25088, Y: 1, X: 1, R: 1, S: 1 }
}

Layer classifier_3 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 4096, C: 4096, Y: 1, X: 1, R: 1, S: 1 }
}

Layer classifier_6 { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1000, C: 4096, Y: 1, X: 1, R: 1, S: 1 }
}

}
