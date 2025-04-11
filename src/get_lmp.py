import requests
import os
from datetime import datetime
import argparse

def get_pjm_data(row_count, start_row, zone, start_date, end_date, output_format, api_key):
    # 构建API URL
    base_url = "https://api.pjm.com/api/v1/da_hrl_lmps"
    
    # 构建查询参数
    params = {
        "RowCount": row_count,
        "startRow": start_row,
        "zone": zone,
        "datetime_beginning_utc": f"{start_date} to {end_date}",
        "format": output_format
    }
    
    # 添加API密钥到请求头
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }
    
    # 发送GET请求
    response = requests.get(base_url, params=params, headers=headers)
    
    # 检查响应状态
    if response.status_code == 200:
        # 确定文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = "csv" if output_format.lower() == "csv" else "json"
        filename = f"pjm_data_{zone}_{current_time}.{file_extension}"
        
        # 保存响应内容到文件
        with open(filename, "wb") as file:
            file.write(response.content)
        
        print(f"数据已成功保存到: {filename}")
        return filename
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")
        return None

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="PJM API数据获取工具",
        epilog="""
示例用法:
1. 使用默认参数: 
   python pjm_api.py
   
2. 指定参数: 
   python pjm_api.py --row_count=10000 --zone=COMED --start_date="1-09-2024 00:00" --end_date="1-09-2024 23:59"
   
3. 更改输出格式为JSON: 
   python pjm_api.py --format=json
   
注意: 日期格式应为 "DD-MM-YYYY HH:MM"
      """
    )
    
    parser.add_argument("--row_count", type=int, default=200000, 
                        help="要获取的最大数据行数 (默认: 200000)")
    
    parser.add_argument("--start_row", type=int, default=1, 
                        help="数据起始行 (默认: 1)")
    
    parser.add_argument("--zone", type=str, default="COMED", 
                        help="PJM电力区域代码 (默认: COMED)")
    
    parser.add_argument("--start_date", type=str, default="1-09-2024 00:00", 
                        help="开始日期时间，格式: 'DD-MM-YYYY HH:MM' (默认: '1-09-2024 00:00')")
    
    parser.add_argument("--end_date", type=str, default="1-09-2024 23:59", 
                        help="结束日期时间，格式: 'DD-MM-YYYY HH:MM' (默认: '1-09-2024 23:59')")
    
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"], 
                        help="输出数据格式 (默认: csv)")
    
    parser.add_argument("--api_key", type=str, default="aca459ab4b064b9ca31ca87ceaa23254", 
                        help="PJM API订阅密钥 (默认: aca459ab4b064b9ca31ca87ceaa23254)")
    
    args = parser.parse_args()
    
    print(f"正在获取PJM数据，区域: {args.zone}，日期范围: {args.start_date} 到 {args.end_date}")
    
    # 调用函数获取数据
    get_pjm_data(
        args.row_count, 
        args.start_row, 
        args.zone, 
        args.start_date, 
        args.end_date, 
        args.format,
        args.api_key
    )