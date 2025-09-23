import hmac
import hashlib
import urllib.parse
import datetime
import unicodedata
import re
from typing import Dict, Optional

class VNPayClient:
    """
    Client tạo URL thanh toán & xác thực phản hồi cho VNPay (Version 2.1.0)
    - Nhận trực tiếp amount_vnd (VND thật, >= 5000)
    - Ký bằng HMAC SHA512 (vnp_SecureHashType=HMACSHA512)
    - hashData: CHUỖI CHƯA URL-ENCODE: key1=value1&key2=value2...
    """

    def __init__(self, tmn_code: str, hash_secret: str, payment_url: str, return_url: str):
        self.tmn_code = tmn_code
        self.hash_secret = (hash_secret or "").strip()
        self.payment_url = payment_url
        self.return_url = return_url

    @staticmethod
    def _remove_accents(text: str) -> str:
        if not text:
            return "Payment"
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^A-Za-z0-9 \-\.]", " ", text)
        text = " ".join(text.split())
        return text[:255]

    @staticmethod
    def _hmac_sha512(secret_key: str, raw_data: str) -> str:
        return hmac.new(secret_key.encode("utf-8"), raw_data.encode("utf-8"), hashlib.sha512).hexdigest()

    def build_payment_url(
        self,
        order_id: str,
        amount_vnd: int,
        order_desc: str,
        user_ip: str,
        bank_code: Optional[str] = None,
        locale: str = "vn",
        order_type: str = "other",
        expire_minutes: int = 15
    ) -> str:
        """
        Tạo URL thanh toán.
        order_id: mã duy nhất (vnp_TxnRef)
        amount_vnd: số tiền VND (>=5000)
        order_desc: mô tả (sẽ làm sạch)
        user_ip: IP khách
        bank_code: nếu muốn fix kênh
        locale: 'vn'/'en'
        """
        if amount_vnd is None or int(amount_vnd) < 5000:
            raise ValueError("amount_vnd must be >= 5000")
        if not order_id:
            raise ValueError("order_id is required")
        if locale not in ("vn", "en"):
            locale = "vn"

        now = datetime.datetime.now()
        expire = now + datetime.timedelta(minutes=expire_minutes)

        params = {
            "vnp_Version": "2.1.0",
            "vnp_Command": "pay",
            "vnp_TmnCode": self.tmn_code,
            "vnp_Amount": str(int(amount_vnd) * 100),   # *100 theo chuẩn
            "vnp_CurrCode": "VND",
            "vnp_TxnRef": order_id,
            "vnp_OrderInfo": self._remove_accents(order_desc),
            "vnp_OrderType": order_type,
            "vnp_Locale": locale,
            "vnp_ReturnUrl": self.return_url,
            "vnp_IpAddr": user_ip or "127.0.0.1",
            "vnp_CreateDate": now.strftime("%Y%m%d%H%M%S"),
            "vnp_ExpireDate": expire.strftime("%Y%m%d%H%M%S"),
        }
        if bank_code:
            params["vnp_BankCode"] = bank_code

        # 1. Sắp xếp key
        sorted_items = sorted(params.items())

        # 2. hashData KHÔNG encode
        hash_data = "&".join(f"{k}={v}" for k, v in sorted_items)

        # 3. Tính secure hash
        secure_hash = self._hmac_sha512(self.hash_secret, hash_data)

        # 4. Tạo query string (lúc này mới encode value)
        query = "&".join(f"{k}={urllib.parse.quote_plus(str(v))}" for k, v in sorted_items)
        final_url = (f"{self.payment_url}?{query}"
                     f"&vnp_SecureHashType=HMACSHA512&vnp_SecureHash={secure_hash}")

        # Debug
        print("===== VNPay BUILD DEBUG =====")
        print("Order ID      :", order_id)
        print("Amount VND    :", amount_vnd)
        print("vnp_Amount    :", params["vnp_Amount"])
        print("HashData      :", hash_data)
        print("SecureHash    :", secure_hash)
        print("Final URL     :", final_url)
        print("=============================")

        return final_url

    def verify_response(self, query_params: Dict[str, str]) -> bool:
        """
        Xác thực response/return từ VNPay.
        query_params: dict (request.args.to_dict()).
        """
        received_hash = query_params.get("vnp_SecureHash")
        if not received_hash:
            return False

        data = {
            k: v for k, v in query_params.items()
            if k not in ("vnp_SecureHash", "vnp_SecureHashType")
               and k.startswith("vnp_")
        }
        sorted_items = sorted(data.items())
        raw_hash_data = "&".join(f"{k}={v}" for k, v in sorted_items)
        calculated = self._hmac_sha512(self.hash_secret, raw_hash_data)

        print("----- VNPay VERIFY DEBUG -----")
        print("HashData Raw :", raw_hash_data)
        print("Received     :", received_hash)
        print("Calculated   :", calculated)
        print("------------------------------")

        return calculated == received_hash


# Convenience functions (nếu muốn dùng theo phong cách cũ)
def build_vnpay_url(order_id, amount_vnd, order_desc, config, user_ip):
    client = VNPayClient(
        tmn_code=config["VNPAY_TMN_CODE"],
        hash_secret=config["VNPAY_HASH_SECRET"],
        payment_url=config["VNPAY_URL"],
        return_url=config["VNPAY_RETURN_URL"]
    )
    return client.build_payment_url(
        order_id=order_id,
        amount_vnd=amount_vnd,
        order_desc=order_desc,
        user_ip=user_ip
    )

def verify_vnpay_response(params, config):
    client = VNPayClient(
        tmn_code=config["VNPAY_TMN_CODE"],
        hash_secret=config["VNPAY_HASH_SECRET"],
        payment_url=config["VNPAY_URL"],
        return_url=config["VNPAY_RETURN_URL"]
    )
    return client.verify_response(params)