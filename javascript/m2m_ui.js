function submit_mov2mov() {
    rememberGallerySelection('mov2mov_gallery')
    showSubmitButtons('mov2mov', false)
    showResultVideo('mov2mov', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('mov2mov_gallery_container'), gradioApp().getElementById('mov2mov_gallery'), function () {
        showSubmitButtons('mov2mov', true)
        showResultVideo('mov2mov', true)
    })

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

function showResultVideo(tabname, show) {
    gradioApp().getElementById(tabname + '_video').style.display = show ? "block" : "none"
    gradioApp().getElementById(tabname + '_gallery').style.display = show ? "none" : "block"

}


function showModnetModels() {
    var check = arguments[0]
    gradioApp().getElementById('mov2mov_modnet_model').style.display = check ? "block" : "none"
    return []
}