
function openModal(imageSrc) {
    let modal = document.getElementById("imageModal");
    let modalImg = document.getElementById("modalImg");
    modalImg.src = imageSrc;
    modal.style.display = "flex";
}

function closeModal() {
    let modal = document.getElementById("imageModal");
    modal.classList.remove("fade-in");
    setTimeout(() => {
        modal.style.display = "none";
    }, 200); // 让动画结束后隐藏
}

// 监听键盘事件，按 ESC 关闭 modal
document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
        closeModal();
    }
});


function predictImage(imagePath) {
    console.log(imagePath)
    fetch(`/predict/${encodeURIComponent(imagePath)}`)
        .then(response => response.json())
        .then(data => showMessage(data.result, data.result === "正例"))
        .catch(error => showMessage("预测失败:" + error, false));
}

function showMessage(message, isSuccess = true) {
    let messageBox = document.getElementById("messageBox");
    messageBox.innerText = message;
    messageBox.style.backgroundColor = isSuccess ? "#4CAF50" : "#f44336"; // 绿 ✔ / 红 ✖
    messageBox.style.display = "block";
    messageBox.style.opacity = "1";

    setTimeout(() => {
        messageBox.style.opacity = "0";
        setTimeout(() => messageBox.style.display = "none", 1000);
    }, 3000);
}
