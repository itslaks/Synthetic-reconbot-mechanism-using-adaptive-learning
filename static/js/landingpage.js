window.onscroll = function() {stickNavbar()};

const navbar = document.getElementById("navbar");
const sticky = navbar.offsetTop;

function stickNavbar() {
  if (window.pageYOffset > sticky) {
    navbar.classList.add("sticky");
  } else {
    navbar.classList.remove("sticky");
  }
}

document.addEventListener('DOMContentLoaded', () => {       
  const message = document.querySelector('.message');
  const button = document.querySelector('.animated-button');
  
  // Message will appemessar first
  message.style.animation = 'slideIn 2s forwards'; 

  // Button will appear after the message animation is complete
  button.style.animation = 'buttonSlideIn 2s forwards 2s'; // Delay button animation by 2s (after message finishes)
});



const particleContainer = document.querySelector('.particles');
const starBackground = document.querySelector('.star-background');

for (let i = 0; i < 100; i++) {
  const particle = document.createElement('div');
  particle.classList.add('particle');
  particle.style.top = ${Math.random() * 100}vh;
  particle.style.left = ${Math.random() * 100}vw;
  particle.style.animationDelay = ${Math.random() * 10}s;
  particleContainer.appendChild(particle);
}

for (let i = 0; i < 300; i++) {
  const star = document.createElement('div');
  star.classList.add('star');
  star.style.top = ${Math.random() * 100}vh;
  star.style.left = ${Math.random() * 100}vw;
  starBackground.appendChild(star);
}

// Slider functionality
const slides = document.querySelector('.slides');
const slide = document.querySelectorAll('.slide');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

let index = 0;

function showSlide(n) {
  index += n;
  if (index >= slide.length) {
    index = 0;
  }
  if (index < 0) {
    index = slide.length - 1;
  }
  slides.style.transform = 'translateX(' + (-index * 100) + '%)';
}

prevBtn.addEventListener('click', () => showSlide(-1));
nextBtn.addEventListener('click', () => showSlide(1));