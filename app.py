
import streamlit as st
import cv2
import numpy as np
import utlis
import tempfile
import time
import base64


curveList = []
avgVal = 10

def getLaneCurve(img, display=2):
    imgcopy = img.copy()
    imgResult = img.copy()
    imgThres = utlis.thresholding(img)
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgwarp = utlis.warpimg(imgThres, points, wT, hT)
    imgwarppoints = utlis.drawPoints(imgcopy, points)
    middlepoint, imghist = utlis.getHistogram(imgwarp, display=True, minPer=0.5, region=4)
    curveAveragepoint, imghist = utlis.getHistogram(imgwarp, display=True, minPer=0.9)
    curveRaw = curveAveragepoint - middlepoint
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))
    if display != 0:
        imgInvWarp = utlis.warpimg(imgwarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgwarppoints, imgwarp],
                                             [imghist, imgLaneColor, imgResult]))
        return imgStacked, curve
    elif display == 1:
        return imgResult, curve

    curve = curve / 100
    if curve > 1: curve == 1
    if curve < -1: curve == -1

    return imgResult, curve

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true" style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )



def main():
    st.title("Road Lane Detection")
    st.write("Upload a video to detect lanes.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        if st.button('Start Video'):
            cap = cv2.VideoCapture(temp_file.name)
            initialTrackBarVals = [174, 85, 44, 225]
            utlis.initializeTrackbars(initialTrackBarVals)
            frame_counter = 0
            left_audio_played = False
            straight_audio_played = False
            right_audio_played = False

            stframe = st.empty()
            stmsg = st.empty()
            
            while cap.isOpened():
                frame_counter += 1
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frame_counter:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                
                success, img = cap.read()
                if not success:
                    break
                
                img = cv2.resize(img, (480, 240))
                img_result, curve = getLaneCurve(img, display=1)
                
                # Check if audio has been played and curve is less than -30
                if not left_audio_played and curve < -30:
                    message = "Turn left"
                    autoplay_audio("left.mp3")
                    left_audio_played = True
                elif curve >= -30 and left_audio_played:
                    left_audio_played = False  # Reset flag if curve is out of range
                
                # Check if audio has been played and curve is between -30 and 30
                if not straight_audio_played and -30 <= curve <= 30:
                    message = "Go straight"
                    autoplay_audio("straight.mp3")
                    straight_audio_played = True
                elif curve < -30 or curve > 30:
                    straight_audio_played = False  # Reset flag if curve is out of range
                    
                # Check if audio has been played and curve is greater than 30
                if not right_audio_played and curve > 30:
                    message = "Turn right"
                    autoplay_audio("right.mp3")
                    right_audio_played = True
                elif curve <= 30 and right_audio_played:
                    right_audio_played = False  # Reset flag if curve is out of range
    
                stframe.image(img_result, channels="BGR")
                stmsg.write(message)



                # Reduce frame rate for better display in Streamlit
                time.sleep(0.03)  # approximately 30 frames per second

            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
