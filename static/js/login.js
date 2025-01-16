console.log("login.js loaded and executing");

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCnVMY7MOrR4s3oafIq05ktG5OCyolXyJQ",
  authDomain: "reconbot-8f5ac.firebaseapp.com",
  projectId: "reconbot-8f5ac",
  storageBucket: "reconbot-8f5ac.appspot.com",
  messagingSenderId: "156458243689",
  appId: "1:156458243689:web:a96ac86d5b4d98d377f4f8"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

document.addEventListener('DOMContentLoaded', function() {
  const submit = document.getElementById('submit');
  if (submit) {
    submit.addEventListener("click", function(event){
      event.preventDefault();  // Prevent form submission
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider)
      .then((result) => {
        const user = result.user;
        console.log(user);
        window.location.href = "/landing";
      }).catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;
        console.error("Login error:", errorMessage);
        // You can add code here to display the error to the user
      });
    });
  } else {
    console.error("Submit button not found");
  }
});

// The following code will run on the landing page and homepage

window.onscroll = function() {stickNavbar()};

const navbar = document.getElementById("navbar");
const sticky = navbar ? navbar.offsetTop : 0;

function stickNavbar() {
  if (navbar) {
    if (window.pageYOffset > sticky) {
      navbar.classList.add("sticky");
    } else {
      navbar.classList.remove("sticky");
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const message = document.querySelector('.message');
  const button = document.querySelector('.animated-button');
  
  if (message) message.style.animation = 'slideIn 2s forwards';
  if (button) button.style.animation = 'buttonSlideIn 2s forwards';

  const particleContainer = document.querySelector('.particles');
  const starBackground = document.querySelector('.star-background');

  if (particleContainer) {
    for (let i = 0; i < 100; i++) {
      const particle = document.createElement('div');
      particle.classList.add('particle');
      particle.style.top = `${Math.random() * 100}vh`;
      particle.style.left = `${Math.random() * 100}vw`;
      particle.style.animationDelay = `${Math.random() * 10}s`;
      particleContainer.appendChild(particle);
    }
  }

  if (starBackground) {
    for (let i = 0; i < 300; i++) {
      const star = document.createElement('div');
      star.classList.add('star');
      star.style.top = `${Math.random() * 100}vh`;
      star.style.left = `${Math.random() * 100}vw`;
      starBackground.appendChild(star);
    }
  }

  // Slider functionality
  const slides = document.querySelector('.slides');
  const slide = document.querySelectorAll('.slide');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  if (slides && slide.length > 0 && prevBtn && nextBtn) {
    let index = 0;

    function showSlide(n) {
      index += n;
      if (index >= slide.length) {
        index = 0;
      }
      if (index < 0) {
        index = slide.length - 1;
      }
      slides.style.transform = `translateX(${-index * 100}%)`;
    }

    prevBtn.addEventListener('click', () => showSlide(-1));
    nextBtn.addEventListener('click', () => showSlide(1));
  }


  }
);