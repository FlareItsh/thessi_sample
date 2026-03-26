import pathlib, textwrap
path = pathlib.Path(r'c:/Projects/thessi_sample/templates/index.html')
lines = path.read_text().splitlines()
new_lines = []
in_script = False
for line in lines:
    if '<script>' in line and not in_script:
        new_lines.append(line)
        in_script = True
        # insert new script body
        new_lines.extend(textwrap.dedent('''
            // DOM Elements
            const fileInput = document.getElementById("fileInput");
            const imagePreview = document.getElementById("imagePreview");
            const previewPlaceholder = document.getElementById("previewPlaceholder");
            const analyzeBtn = document.getElementById("analyzeBtn");
            const analyzeBtnText = document.getElementById("analyzeBtnText");
            const analyzeBtnBgHover = document.getElementById("analyzeBtnBgHover");
            const resultContainer = document.getElementById("resultContainer");
            const resultText = document.getElementById("resultText");
            const loadingIcon = document.getElementById("loadingIcon");

            // Camera Elements
            const video = document.getElementById("video");
            const cameraContainer = document.getElementById("cameraContainer");
            const cameraBtnText = document.getElementById("cameraBtnText");
            const captureCanvas = document.getElementById("captureCanvas");

            let stream = null;
            let selectedFile = null;
            let base64Image = null;
            let lastSpots = [];

            function enableAnalyzeBtn() {
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove("from-gray-700", "to-gray-600");
                analyzeBtn.class.add("from-primary", "to-purple-600");
                analyzeBtnBgHover.classList.remove("hidden");
            }

            // File upload
            function previewFile() {
                const file = fileInput.files[0];
                if (file) {
                    selectedFile = file;
                    base64Image = null;
                    const reader = new FileReader();
                    reader.onload = e => showPreview(e.target.result);
                    reader.readAsDataURL(file);
                    if (stream) stopCamera();
                }
            }

            // camera
            async function toggleCamera() {
                if (stream) return stopCamera();
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
                    video.srcObject = stream;
                    cameraContainer.classList.remove("hidden");
                    cameraBtnText.textContent = "Close Camera";
                } catch (err) {
                    console.error(err);
                    alert("Could not access camera. Please allow permissions or upload a file.");
                }
            }
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(t=>t.stop());
                    stream = null;
                    cameraContainer.classList.add("hidden");
                    cameraBtnText.textContent = "Open Camera";
                }
            }
            function captureImage() {
                if (!stream) return;
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
                const ctx = captureCanvas.getContext("2d");
                ctx.drawImage(video,0,0,captureCanvas.width,captureCanvas.height);
                const dataUrl = captureCanvas.toDataURL("image/jpeg",0.9);
                base64Image = dataUrl;
                selectedFile = null;
                showPreview(dataUrl);
                stopCamera();
            }

            function drawOverlay() {
                const canvas = document.getElementById("overlay");
                const ctx = canvas.getContext("2d");
                canvas.width = imagePreview.clientWidth;
                canvas.height = imagePreview.clientHeight;
                ctx.clearRect(0,0,canvas.width,canvas.height);
                if(!lastSpots||!lastSpots.length) return;
                const sx = canvas.width/imagePreview.naturalWidth;
                const sy = canvas.height/imagePreview.naturalHeight;
                ctx.strokeStyle='lime';ctx.lineWidth=2;
                lastSpots.forEach(([x,y,r])=>{
                    ctx.beginPath();ctx.arc(x*sx,y*sy,r*sx,0,2*Math.PI);ctx.stroke();
                });
            }

            function showPreview(src){
                imagePreview.onload=()=>drawOverlay();
                imagePreview.src=src;
                imagePreview.classList.remove("hidden");
                previewPlaceholder.classList.add("hidden");
                resultContainer.classList.add("hidden");
                enableAnalyzeBtn();
            }

            async function analyzeImage(){
                if(!selectedFile&&!base64Image) return;
                analyzeBtn.disabled=true;analyzeBtnText.textContent="Analyzing...";
                loadingIcon.classList.remove("hidden");resultContainer.classList.add("hidden");
                try{
                    let fd=new FormData();
                    if(selectedFile)fd.append("file",selectedFile);
                    else if(base64Image)fd.append("image_base64",base64Image);
                    const resp=await fetch("/predict",{method:"POST",body:fd});
                    const data=await resp.json();
                    if(resp.ok){
                        lastSpots=data.spots||[];
                        showResult(data.result);
                        drawOverlay();
                    } else{
                        showResult(data.error||"An error occurred",true);
                    }
                }catch(e){console.error(e);showResult("Failed to connect to server.",true);}finally{
                    analyzeBtn.disabled=false;analyzeBtnText.textContent="Analyze Another Image";
                    loadingIcon.classList.add("hidden");
                }
            }

            function showResult(message,isError=false){
                resultContainer.classList.remove("hidden","border-red-500/50","bg-red-500/10","border-green-500/50","bg-green-500/10","border-rose-500/50","bg-rose-500/10");
                if(isError){
                    resultContainer.classList.add("border-red-500/50","bg-red-500/10");
                    resultText.className="text-xl font-bold text-red-400";
                }else if(message.includes("Detected")){
                    resultContainer.classList.add("border-rose-500/50","bg-rose-500/10");
                    resultText.className="text-3xl font-bold text-rose-400";
                }else{
                    resultContainer.classList.add("border-green-500/50","bg-green-500/10");
                    resultText.className="text-3xl font-bold text-green-400";
                }
                resultText.textContent=message;
                const m=message.match(/(\d+) spots?/i);
                if(m){
                    const prev=resultContainer.querySelector(".subcount");if(prev)prev.remove();
                    const sub=document.createElement("p");
                    sub.className="subcount text-sm text-gray-300 mt-1";
                    sub.textContent=`Detected ${m[1]} acne spot${m[1]!="1"?"s":""}`;
                    resultContainer.appendChild(sub);
                }
                resultContainer.classList.add("scale-105");
                setTimeout(()=>resultContainer.classList.remove("scale-105"),200);
            }
        ''').splitlines()
        continue
    if '</script>' in line and in_script:
        new_lines.append(line)
        in_script=False
        continue
    if in_script:
        continue
    new_lines.append(line)
path.write_text("\n".join(new_lines))
print("rewrote script")
