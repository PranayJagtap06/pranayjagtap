// // Initialize EmailJS
// (function () {
//     emailjs.init({publicKey: "o8aRWsvB31ul6z2bB",}); // Replace with your EmailJS user ID
// })();

// async function handleSubmit(event) {
//     event.preventDefault();

//     const formData = {
//         name: document.getElementById('name').value,
//         phone: document.getElementById('phone').value,
//         email: document.getElementById('email').value,
//         company: document.getElementById('company').value,
//         subject: document.getElementById('subject').value,
//         message: document.getElementById('message').value,
//     };

//     // Validate email
//     const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
//     if (!emailRegex.test(formData.email)) {
//         showNotification('error', 'Invalid email address');
//         return;
//     }

//     try {
//         const response = await emailjs.send('service_hd12oej', 'template_kjcv6ia', formData);
//         showNotification('success', 'Message sent successfully!');
//     } catch (error) {
//         showNotification('error', 'Error sending message: ' + error.text);
//     }
// }

function showNotification(type, message) {
    const notificationContainer = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerText = message;
    notificationContainer.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 5000);
}


// install: npm install express body-parser dotenv emailjs
async function handleSubmit(event) {
    event.preventDefault();

    const formData = {
        name: document.getElementById('name').value,
        phone: document.getElementById('phone').value,
        email: document.getElementById('email').value,
        company: document.getElementById('company').value,
        subject: document.getElementById('subject').value,
        message: document.getElementById('message').value,
    };

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
        showNotification('error', 'Invalid email address');
        return;
    }

    try {
        const response = await fetch('/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        });

        const data = await response.json();
        if (response.ok) {
            showNotification('success', data.message);
        } else {
            showNotification('error', data.message);
        }
    } catch (error) {
        showNotification('error', 'Error sending message: ' + error.message);
    }
}
