const slider = document.querySelector('.slider');

function activate(e) {
    const items = document.querySelectorAll('.item');
    if (e.target.matches('.next')) {
        slider.append(items[0]);
        updateActiveStates();
    }
    if (e.target.matches('.prev')) {
        slider.prepend(items[items.length-1]);
        updateActiveStates();
    }
}

function updateActiveStates() {
    const items = document.querySelectorAll('.item');
    items.forEach((item, index) => {
        if (index < 2) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

document.addEventListener('click', activate, false);
// Initialize active states on page load
document.addEventListener('DOMContentLoaded', updateActiveStates);
