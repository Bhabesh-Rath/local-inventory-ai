# Where's My Stuff? - A Smart Inventory App

## Overview

"Where's My Stuff?" is a mobile application designed to help you keep track of your belongings. Using your phone's camera, you can quickly scan items, and the app will automatically identify them using a sophisticated machine-learning model. You can then assign your items to a specific place, making it easy to remember where you've stored them.

The app features a clean, modern interface with both light and dark themes, and it's built with simplicity and ease of use in mind.

## Features

*   **AI-Powered Object Recognition:** The app uses a pre-trained TensorFlow Lite model to automatically identify and suggest labels for your items with a confidence score of 25% or higher.
*   **Camera-Based Scanning:** Simply point your camera at an item and take a picture. The app handles the rest.
*   **Multi-Label Selection:** After scanning, you can select multiple accurate labels from a list of suggestions.
*   **Place-Based Organization:** Assign your scanned items to a "place" (e.g., "Garage," "Kitchen Cupboard") to keep your inventory organized.
*   **Persistent Inventory:** Your inventory is saved locally on your device, so you can access it anytime.
*   **Theme Customization:** Choose between a light or dark theme, and the app will remember your preference.
*   **Modern, Responsive UI:** The app is built with Flutter and Material 3, providing a beautiful and intuitive user experience on any device.

## Technical Details

*   **Framework:** Flutter 3.x
*   **Language:** Dart 3.x
*   **State Management:** `flutter_riverpod` for efficient and scalable state management.
*   **Database:** `sqlite3` and `sqlite3_flutter_libs` for reliable local data storage.
*   **Machine Learning:** `tflite_flutter` for running the on-device TensorFlow Lite model.
*   **Camera:** `camera` and `camera_android_camerax` for seamless camera integration.
*   **UI:** Material 3 design principles for a modern and consistent look and feel.

## How to Build and Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    ```
2.  **Get dependencies:**
    ```bash
    flutter pub get
    ```
3.  **Run the app:**
    ```bash
    flutter run
    ```

## Screenshots

*(You can add screenshots of the app here to showcase its features. For example, a screenshot of the camera view, the label selection dialog, and the inventory list.)*

**Example Screenshot Placeholder:**

![App Screenshot](https://via.placeholder.com/300x600.png?text=App+Screenshot)

## Future Improvements

*   **Cloud Sync:** Sync your inventory across multiple devices using a cloud-based backend like Firebase.
*   **Search Functionality:** Add a search bar to quickly find items in your inventory.
*   **Barcode/QR Code Scanning:** Extend the app's capabilities to include barcode and QR code scanning for even faster item identification.
*   **Custom Labels:** Allow users to add their own custom labels for items that the AI model may not recognize.
