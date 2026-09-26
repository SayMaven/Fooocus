// From A1111 / Fooocus Lightbox Modal

function closeModal() {
    const lb = gradioApp().getElementById("lightboxModal");
    if (lb) lb.style.display = "none";
}

function showModal(event) {
    const source = event.target || event.srcElement;
    const modalImage = gradioApp().getElementById("modalImage");
    const lb = gradioApp().getElementById("lightboxModal");
    if (!modalImage || !lb) return;

    const imgSrc = source.src || source.querySelector('img')?.src;
    if (!imgSrc) return;

    modalImage.src = imgSrc;
    if (modalImage.style.display === 'none') {
        lb.style.setProperty('background-image', 'url(' + imgSrc + ')');
    }
    lb.style.display = "flex";
    lb.focus();

    event.stopPropagation();
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = gradioApp().getElementById("modalImage");
    if (modalImage && modalImage.offsetParent) {
        let currentButton = selected_gallery_button();
        let currentImg = currentButton?.querySelector('img') || currentButton?.children?.[0];

        if (currentImg && currentImg.src && modalImage.src != currentImg.src) {
            modalImage.src = currentImg.src;
            if (modalImage.style.display === 'none') {
                const modal = gradioApp().getElementById("lightboxModal");
                modal.style.setProperty('background-image', `url(${modalImage.src})`);
            }
        }
    }
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('#final_gallery .grid-container button, #final_gallery button, .image_gallery .thumbnails > .thumbnail-item, .image_gallery button, #final_gallery [data-testid*="thumbnail"], #final_gallery [aria-label*="Thumbnail"], #final_gallery .thumbnail-item');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if ((elem.offsetParent || elem.parentElement?.offsetParent) && (elem.querySelector('img') || elem.tagName === 'IMG')) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    var buttons = all_gallery_buttons();
    return buttons.find(elem => elem.classList.contains('selected')) || buttons[0] || null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

function modalImageSwitch(offset) {
    var galleryButtons = all_gallery_buttons();

    if (galleryButtons.length > 1) {
        var currentButton = selected_gallery_button();

        var result = -1;
        galleryButtons.forEach(function(v, i) {
            if (v == currentButton) {
                result = i;
            }
        });

        if (result != -1) {
            var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
            nextButton.click();
            const modalImage = gradioApp().getElementById("modalImage");
            const modal = gradioApp().getElementById("lightboxModal");
            const nextImg = nextButton.querySelector('img') || (nextButton.tagName === 'IMG' ? nextButton : nextButton.children?.[0]);
            if (nextImg && nextImg.src && modalImage) {
                modalImage.src = nextImg.src;
                if (modalImage.style.display === 'none' && modal) {
                    modal.style.setProperty('background-image', `url(${modalImage.src})`);
                }
            }
            if (modal) {
                setTimeout(function() {
                    modal.focus();
                }, 10);
            }
        }
    }
}

function saveImage() {

}

function modalSaveImage(event) {
    event.stopPropagation();
}

function modalNextImage(event) {
    modalImageSwitch(1);
    event.stopPropagation();
}

function modalPrevImage(event) {
    modalImageSwitch(-1);
    event.stopPropagation();
}

function modalKeyHandler(event) {
    switch (event.key) {
    case "s":
        saveImage();
        break;
    case "ArrowLeft":
        modalPrevImage(event);
        break;
    case "ArrowRight":
        modalNextImage(event);
        break;
    case "Escape":
        closeModal();
        break;
    }
}

function setupImageForLightbox(e) {
    if (e.dataset.modded) {
        return;
    }

    e.dataset.modded = true;

    var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;

    // For Firefox, listening on click first switched to next image then shows the lightbox.
    var event = isFirefox ? 'mousedown' : 'click';

    e.addEventListener(event, function(evt) {
        // Stage 1 check: If click was on a thumbnail inside the grid, do NOT intercept!
        // Allow Gradio's native handler to expand the thumbnail full onto the canvas first (Stage 1 -> Stage 2)
        if (evt.target.closest('.grid-container') || evt.target.closest('.thumbnail-item') || evt.target.closest('.grid-wrap')) {
            return;
        }

        if (evt.button == 1) {
            open(evt.target.src);
            evt.preventDefault();
            return;
        }
        if (evt.button != 0) return;

        // Stage 2 -> Stage 3: When clicking the image that is already full on the canvas,
        // open the full-screen centered Lightbox Modal!
        modalZoomSet(gradioApp().getElementById('modalImage'), true);
        evt.preventDefault();
        showModal(evt);
    }, true);

}

function modalZoomSet(modalImage, enable) {
    if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
}

function modalZoomToggle(event) {
    var modalImage = gradioApp().getElementById("modalImage");
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
    event.stopPropagation();
}

function modalTileImageToggle(event) {
    const modalImage = gradioApp().getElementById("modalImage");
    const modal = gradioApp().getElementById("lightboxModal");
    const isTiling = modalImage.style.display === 'none';
    if (isTiling) {
        modalImage.style.display = 'block';
        modal.style.setProperty('background-image', 'none');
    } else {
        modalImage.style.display = 'none';
        modal.style.setProperty('background-image', `url(${modalImage.src})`);
    }

    event.stopPropagation();
}

onAfterUiUpdate(function() {
    var fullImg_preview = gradioApp().querySelectorAll('#final_gallery img, .image_gallery img, #progress_gallery img');
    if (fullImg_preview != null) {
        fullImg_preview.forEach(function(img) {
            setupImageForLightbox(img);
            // Give pointer cursor to the expanded canvas preview (Stage 2)
            if (!img.closest('.grid-container') && !img.closest('.thumbnail-item') && !img.closest('.grid-wrap')) {
                img.style.cursor = 'pointer';
            }
        });
    }
    updateOnBackgroundChange();
});

function initModal() {
    if (document.getElementById("lightboxModal")) return;

    const modal = document.createElement('div');
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex = 0;
    modal.addEventListener('keydown', modalKeyHandler, true);

    const modalControls = document.createElement('div');
    modalControls.className = 'modalControls gradio-container';
    modal.append(modalControls);

    const modalZoom = document.createElement('span');
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;';
    modalZoom.addEventListener('click', modalZoomToggle, true);
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom);

    const modalClose = document.createElement('span');
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;';
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose);

    const modalImage = document.createElement('img');
    modalImage.id = 'modalImage';
    modalImage.onclick = closeModal;
    modalImage.tabIndex = 0;
    modalImage.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalImage);

    const modalPrev = document.createElement('a');
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;';
    modalPrev.tabIndex = 0;
    modalPrev.addEventListener('click', modalPrevImage, true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalPrev);

    const modalNext = document.createElement('a');
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;';
    modalNext.tabIndex = 0;
    modalNext.addEventListener('click', modalNextImage, true);
    modalNext.addEventListener('keydown', modalKeyHandler, true);

    modal.appendChild(modalNext);

    try {
        gradioApp().appendChild(modal);
    } catch (e) {
        gradioApp().body.appendChild(modal);
    }

    document.body.appendChild(modal);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initModal);
} else {
    initModal();
}

if (typeof onUiLoaded === "function") {
    onUiLoaded(initModal);
}
