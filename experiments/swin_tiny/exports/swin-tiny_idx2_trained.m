Network swin-tiny {

FPS: 30

Layer swin_embeddings_patch_embeddings_projection { 
	Type: CONV
	Stride { X: 4, Y: 4 }
	Dimensions: { N: 1, K: 96, C: 3, Y: 224, X: 224, R: 4, S: 4 }
}

Layer swin_encoder_layers_0_blocks_0_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_0_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_0_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_0_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_0_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_0_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 64, K: 96, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 96, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_blocks_1_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 96, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_0_downsample_reduction { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_0_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 16, K: 192, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 192, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_blocks_1_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 192, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_1_downsample_reduction { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_0_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_1_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_2_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_3_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_4_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 4, K: 384, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1536, C: 384, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_blocks_5_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 384, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_2_downsample_reduction { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 1536, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_intermediate_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 3072, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_0_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 3072, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_attention_self_query { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_attention_self_key { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_attention_self_value { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_attention_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_intermediate_dense_d2s_module_conv { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 49, K: 768, C: 192, Y: 2, X: 2, R: 1, S: 1 }
}

Layer swin_encoder_layers_3_blocks_1_output_dense { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 768, C: 3072, Y: 1, X: 1, R: 1, S: 1 }
}

Layer classifier { 
	Type: CONV
	Stride { X: 1, Y: 1 }
	Dimensions: { N: 1, K: 1000, C: 768, Y: 1, X: 1, R: 1, S: 1 }
}

}
