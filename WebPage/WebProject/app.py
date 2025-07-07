from flask import Flask, render_template, request, redirect, jsonify
import os

app = Flask(__name__)

click_index = {
    "stage": 1  # 전체 클릭 수로 통합 관리
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 업로드한 파일을 static/previews 폴더에 저장
    request.files['front_img'].save(os.path.join('static/previews', 'front.png'))
    request.files['top_img'].save(os.path.join('static/previews', 'top.png'))
    click_index["stage"] = 1  # TOP 이미지부터 시작
    return redirect('/segment')

@app.route('/segment')
def segment():
    stage = click_index["stage"]

    # 이미지 파일 및 상태 메시지 구성
    if 1 <= stage <= 6:
        # TOP 관련 이미지 표시 (top_1.png, top_2.png, ...)
        img_file = f"top_{stage - 1}"
        view_name = "TOP View"
        status_msg = f"TOP {stage - 1}번째 객체 선택 중"
    elif stage == 7:
        # TOP 세그먼트 완료 후 FRONT 이미지들로 자동 전환 (same page transition)
        img_file = f"front_0"
        view_name = "FRONT View"
        status_msg = "FRONT 원본 이미지 표시 중"
    elif 8 <= stage <= 12:
        # FRONT 관련 이미지 표시 (front_1.png, front_2.png, ...)
        img_file = f"front_{stage - 8}"
        view_name = "FRONT View"
        status_msg = f"FRONT {stage - 8}번째 객체 선택 중"
    elif stage == 13:
        # FRONT 세그먼트 완료 후 결과 페이지로 이동 (results.html로 리디렉션)
        return render_template('results.html')  # results.html로 리디렉션
    else:
        # 모든 단계가 끝난 후 결과 페이지로 리디렉션
        return redirect('/result')

    return render_template('segment.html',
                           img_file=img_file,  # 이미지 파일 전달
                           view_name=view_name,
                           status_msg=status_msg)


@app.route('/click', methods=['POST'])
def click():
    stage = click_index["stage"]

    # 마지막 이미지를 클릭하면 바로 results로 리디렉션
    if stage == 13:
        return redirect('/result')  # results 페이지로 리디렉션

    click_index["stage"] += 1  # 클릭할 때마다 단계 진행
    return jsonify({'redirect': '/segment'})  # segment 페이지로 리디렉션

@app.route('/result')
def result():
    return render_template('results.html')

@app.route('/transform', methods=['POST'])
def transform():
    return render_template('transform_result.html', result_img='item_model.png')

@app.route('/optimize')
def optimize():
    # /optimize 경로로 이동하면 박스를 구매할 장소를 선택하는 화면을 보여준다
    return render_template('optimize.html')  # optimize.html 템플릿을 렌더링

@app.route('/show_final_result')
def show_final_result():
    # final_result.html로 리디렉션
    return render_template('final_result.html')  # final_result.html 템플릿을 렌더링


if __name__ == '__main__':
    app.run(debug=True)
