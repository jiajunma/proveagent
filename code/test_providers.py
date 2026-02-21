#!/usr/bin/env python3
"""
测试所有可用的模型提供商连接
自动检测环境变量并测试配置好的 API
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_providers import (
    GeminiProvider, 
    OpenAIProvider, 
    KimiProvider,
    OPENAI_SDK_AVAILABLE
)

# 测试用的简单问题
TEST_PROMPT = "计算 2+2 等于多少？只回答数字即可。"
SYSTEM_PROMPT = "你是一个数学助手，请简洁回答。"


def test_provider(provider_class, provider_name, model_name, env_var):
    """测试单个提供商"""
    print(f"\n{'='*60}")
    print(f"测试 {provider_name}: {model_name}")
    print(f"{'='*60}")
    
    # 检查环境变量
    api_key = os.getenv(env_var)
    if not api_key:
        print(f"⏭️  跳过: {env_var} 环境变量未设置")
        return None
    
    print(f"✅ {env_var} 已设置")
    
    # 特殊检查：Kimi 需要 OpenAI SDK
    if provider_name == "Kimi" and not OPENAI_SDK_AVAILABLE:
        print("❌ OpenAI SDK 未安装")
        print("   请运行: pip install openai>=1.0.0")
        return False
    
    try:
        # 创建 provider
        provider = provider_class(api_key=api_key, model_name=model_name)
        
        # 检查能力
        supports_streaming = provider.check_capabilities()
        print(f"   流式支持: {supports_streaming}")
        
        # 构建请求 - 强制使用 temperature=1.0
        payload = provider.build_request_payload(
            system_prompt=SYSTEM_PROMPT,
            question_prompt=TEST_PROMPT,
            other_prompts=None,
            enable_thinking=True,
            streaming=False
        )
        # 强制覆盖温度为 1.0（不同 provider 格式不同）
        if provider_name == "Gemini":
            if 'generationConfig' not in payload:
                payload['generationConfig'] = {}
            payload['generationConfig']['temperature'] = 1.0
            display_temp = payload['generationConfig'].get('temperature', 1.0)
        else:
            payload['temperature'] = 1.0
            display_temp = payload.get('temperature', 1.0)
        
        print(f"   模型: {payload.get('model', model_name)}")
        print(f"   温度: {display_temp} (强制设置)")
        
        # 发送请求
        print(f"   发送请求...")
        response = provider.send_api_request(payload, streaming=False, show_thinking=False)
        
        # 提取响应
        text, thinking = provider.extract_text_from_response(response)
        
        if text:
            print(f"✅ 连接成功!")
            print(f"   完整响应 (长度: {len(text)}):")
            print(f"   {'-'*50}")
            print(f"   {text}")
            print(f"   {'-'*50}")
            
            # 显示 thinking 信息
            if thinking:
                print(f"   完整 Thinking (长度: {len(thinking)}):")
                print(f"   {'-'*50}")
                print(f"   {thinking}")
                print(f"   {'-'*50}")
            else:
                # 说明为什么没有 thinking
                if provider_name == "OpenAI":
                    print(f"   ℹ️  OpenAI 模型不支持 thinking 功能")
                elif provider_name == "Gemini":
                    print(f"   ℹ️  Gemini 在此模式下不返回 thinking")
                elif provider_name == "Kimi":
                    print(f"   ⚠️  Kimi 应该返回 thinking，但为空")
            return True
        else:
            print(f"⚠️  响应为空 (text 长度为 0)")
            print(f"   原始响应: {response}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("模型提供商连接测试")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenAI SDK: {'✅ 已安装' if OPENAI_SDK_AVAILABLE else '❌ 未安装'}")
    
    results = {}
    
    # 测试 Gemini
    results['Gemini'] = test_provider(
        GeminiProvider, 
        "Gemini", 
        "gemini-2.5-pro",
        "GOOGLE_API_KEY"
    )
    
    # 测试 OpenAI
    results['OpenAI'] = test_provider(
        OpenAIProvider,
        "OpenAI",
        "gpt-4o",
        "OPENAI_API_KEY"
    )
    
    # 测试 Kimi
    results['Kimi'] = test_provider(
        KimiProvider,
        "Kimi",
        "kimi-k2-thinking",
        "KIMI_API_KEY"
    )
    
    # 总结
    print(f"\n{'='*60}")
    print("测试结果总结:")
    print(f"{'='*60}")
    
    available = []
    for name, result in results.items():
        if result is None:
            status = "⏭️  未配置"
        elif result is True:
            status = "✅ 可用"
            available.append(name)
        else:
            status = "❌ 失败"
        print(f"{name:12s}: {status}")
    
    print(f"\n可用提供商: {', '.join(available) if available else '无'}")
    
    # 设置建议
    print(f"\n{'='*60}")
    print("环境变量设置:")
    print(f"{'='*60}")
    if results.get('Gemini') is None:
        print("export GOOGLE_API_KEY='your_gemini_api_key'")
    if results.get('OpenAI') is None:
        print("export OPENAI_API_KEY='your_openai_api_key'")
    if results.get('Kimi') is None:
        print("export KIMI_API_KEY='your_kimi_api_key'")
    
    # 返回码
    if available:
        print(f"\n🎉 至少有一个提供商可用!")
        return 0
    else:
        print(f"\n⚠️ 没有可用的提供商")
        return 1


if __name__ == "__main__":
    sys.exit(main())
