/**
 * 引入即注册。新增算子只需在此加一行 import。
 *
 * v0 已实现：attention/mha、ffn-moe/dense-ffn、ffn-moe/deepseek-moe
 * 后续：attention/mla、attention/flash（作为 Implementation）、kvcache/*
 */
import './attention';
import './ffn-moe';
