let trainvisible = false
function clickTrain() {
    let block = document.getElementById("traincode")
    let revhidetext = document.getElementById("revhide")
    if (!trainvisible) {
        block.style.display = "block"
        revhidetext.innerHTML = "hide"
    } else {
        block.style.display = "none"
        revhidetext.innerHTML = "reveal"
    }
    trainvisible = !trainvisible;
}