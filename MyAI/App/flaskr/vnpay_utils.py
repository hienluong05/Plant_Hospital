import hmac
import hashlib
import urllib.parse
import datetime
import requests

# Hàm lấy tỷ giá USD hiện tại (USD -> VND) từ exchangerate.host
def get_usd_to_vnd():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=VND"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data["rates"]["VND"])
    except Exception:
        # Nếu lỗi, trả về tỷ giá mặc định
        return 25000.0

# Hàm chuyển đổi USD sang VND
def usd_to_vnd(amount_usd):
    rate = get_usd_to_vnd()
    return int(round(amount_usd * rate))

def build_vnpay_url(order_id, amount_usd, order_desc, config, user_ip):
    # Chuyển đổi sang VND (nếu amount_usd là USD)
    amount_vnd = usd_to_vnd(amount_usd)

    now = datetime.datetime.now()
    expire = now + datetime.timedelta(minutes=15)
    params = {
        'vnp_Version': '2.1.0',
        'vnp_Command': 'pay',
        'vnp_TmnCode': config['VNPAY_TMN_CODE'],
        'vnp_Amount': str(amount_vnd * 100),  # VNPay yêu cầu VND * 100
        'vnp_CurrCode': 'VND',
        'vnp_TxnRef': str(order_id),
        'vnp_OrderInfo': order_desc,
        'vnp_OrderType': 'other',
        'vnp_ReturnUrl': config['VNPAY_RETURN_URL'],
        'vnp_IpAddr': user_ip,
        'vnp_Locale': 'vn',
        'vnp_CreateDate': now.strftime('%Y%m%d%H%M%S'),
        'vnp_ExpireDate': expire.strftime('%Y%m%d%H%M%S'),
        # Có thể thêm vnp_BankCode nếu muốn chỉ định loại thanh toán
    }
    
    print(amount_usd, amount_vnd)
    
    # 1. Sắp xếp key tăng dần
    sorted_params = sorted(params.items())
    # 2. Tạo chuỗi hashData (không encode giá trị)
    hash_data = '&'.join([f"{k}={v}" for k, v in sorted_params])
    # 3. Tạo secure hash HMAC SHA256
    key_bytes = config['VNPAY_HASH_SECRET'].encode('utf-8')
    data_bytes = hash_data.encode('utf-8')
    sign = hmac.new(key_bytes, data_bytes, hashlib.sha256).hexdigest()
    # 4. Tạo query string (giá trị phải encode)
    query = '&'.join([f"{k}={urllib.parse.quote_plus(str(v))}" for k, v in sorted_params])
    # 5. Ghép vào URL
    url = f"{config['VNPAY_URL']}?{query}&vnp_SecureHashType=SHA256&vnp_SecureHash={sign}"
    return url

# Ví dụ sử dụng:
# config = {
#     "VNPAY_TMN_CODE": "xxxxxx",
#     "VNPAY_HASH_SECRET": "xxxxxxxxxxxxxxxxxxxxxxxxxxx",
#     "VNPAY_URL": "https://sandbox.vnpayment.vn/paymentv2/vpcpay.html",
#     "VNPAY_RETURN_URL": "http://localhost:5000/vnpay_return"
# }
# url = build_vnpay_url("order123", 45.5, "Thanh toan hoa don 123", config, "127.0.0.1")