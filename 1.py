def show_cumulative_percent_complete(data):
        # 1. 生成1-20的所有键
        all_keys = range(1, 21)
        
        # 2. 计算总个数
        total = sum(data.values())
        
        # 3. 计算累计个数
        cumulative = 0
        result = {}
        
        for key in all_keys:
            cumulative += data.get(key, 0)
            result[key] = cumulative/total
        
        # 4. 生成输出字符串
        keys_str = " ".join(map(str, all_keys))
        values_str = " ".join(result.values())
        
        return f"键值:{keys_str} 累计百分比：{values_str}"

    # 示例使用
data = {20: 5, 18: 1, 19: 5}
print(show_cumulative_percent_complete(data))