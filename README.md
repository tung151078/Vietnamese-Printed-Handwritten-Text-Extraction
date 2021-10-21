# Vietnamese-Printed-Handwritten-Text-Extraction
Bao gồm các file:
- regionselector.py: chạy file này để lựa chọn vùng muốn extract
- main.py: chạy trích xuất text từ các form nằm trong thư mục ‘samples’
- app.py: file build Streamlit
- dataoutput.csv: lưu data

Thư viện sử dụng:
- vietocr : Model của vietOCR được sử dụng để predict chữ viết tiếng Việt.
- vietnam_number: sử dụng để chuyển đổi tiền từ chữ sang số và ngược lại.
- pdf2image: sử dụng để chuyển đổi file từ ‘.pdf’ sang ‘.png’
- matplotlib , numpy, streamlit …

Trước khi chạy file main.py, chạy file regionselector.py để lựa chọn vùng muốn trích xuất.
Khi chạy file, form mẫu sẽ sẽ hiển thị. Chọn vùng muốn trích xuất: ở đây sẽ chọn vùng bằng cách click chuột vào tọa độ phía trên bên trái, sau đó click vào tọa độ phía góc dưới bên phải của vùng muốn lựa chọn. Nhập vào đây là 'text' hay ô 'checkbox': nếu là trên form là text thì nhập là ‘text’, nếu là ô 'checkbox' thì nhập là ‘box’. Ví dụ như bên dưới:

'''
- Enter Type text
- Enter Name ten
- Enter Type text
- Enter Name dia_chi
    
[[(232, 208), (747, 242), 'text', 'ten'], [(120, 240), (746, 271), 'text', 'dia_chi']]

'''

Copy dòng cuối cho vào biến ‘roi’ trong main.py. Những tọa độ đó sẽ được sử dụng để cắt ra đưa vào model để predict.
