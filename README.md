# VOICE-FYP: Bi-Directional Pakistani Sign Language Translation

VOICE-FYP is a real-time, bi-directional communication platform that translates between Pakistani Sign Language (PSL), text, and speech.  
It is designed to reduce communication barriers between Deaf/Hard-of-Hearing and hearing communities through accessible, practical technology.

## Live Deployment

https://voice-fyp-deployed.vercel.app/

## Executive Summary

Communication inequality is one of the most persistent barriers to inclusive education, healthcare, employment, and public services. In Pakistan, where localized sign language support is still limited in mainstream digital systems, this gap is especially visible.

VOICE-FYP addresses this challenge by enabling:

- PSL to text conversion for improved comprehension by non-signers  
- Text and speech pathways for broader interaction and response  
- Integrated bidirectional communication flow in a single platform

The project combines machine learning and modern web engineering to deliver real-time translation workflows that are usable in day-to-day scenarios.

## Problem Statement

Many Deaf and Hard-of-Hearing individuals face routine communication friction in environments where interpreters are unavailable. Existing global solutions are often not adapted to Pakistani Sign Language, local context, or local user needs.

This project was built to contribute a localized, technically grounded step toward communication equity.

## Social Impact

VOICE-FYP aims to create measurable social value in four major dimensions:

1. **Accessibility and Inclusion**  
   It supports more direct interaction between signers and non-signers, reducing dependence on intermediaries for basic communication.

2. **Educational Opportunity**  
   It can assist classrooms and learning environments by enabling clearer instruction exchange across mixed-ability groups.

3. **Healthcare and Public Service Communication**  
   It can improve clarity in sensitive contexts where misunderstandings carry high personal cost.

4. **Workplace Participation**  
   It can help organizations create more inclusive communication channels for Deaf professionals and clients.

Beyond technical performance, the broader impact of VOICE-FYP is its contribution to dignity, autonomy, and equal participation in society.

## Core Capabilities

- Real-time bidirectional translation workflow
- Pakistani Sign Language oriented processing
- Text and speech interoperability
- Unified web interface for multimodal interaction
- Modular architecture for model and application integration

## System Architecture (High Level)

The system is structured around four primary layers:

1. **Frontend Application**  
   Handles user interaction, input capture, and translation display.

2. **Backend / API Layer**  
   Coordinates requests, model inference calls, and response orchestration.

3. **Machine Learning Inference Layer**  
   Processes sign-language input and generates structured textual output.

4. **Speech and Text Processing Layer**  
   Manages speech-to-text and text-to-speech paths in the communication loop.

## Technology Foundation

- Web application frontend
- Backend service/API integration
- Machine learning inference for sign-language translation
- Deployment on Vercel for public accessibility

## Repository Setup

### 1. Clone the repository

```bash
git clone https://github.com/aliyanz85/bi-directional-pakistan-sign-language-translation.git
cd bi-directional-pakistan-sign-language-translation
```

### 2. Install dependencies

Install dependencies for each module in the repository according to its package configuration.

```bash
npm install
```

### 3. Configure environment

Create required environment files (for example, `.env` or `.env.local`) and provide values for:

- API/service endpoints
- model or inference configuration
- speech processing service keys (if applicable)

### 4. Run development servers

Use the scripts defined in each module.

```bash
npm run dev
```

### 5. Access locally

`http://localhost:3000`

## Team

| Name | Roll Number | Contribution Area |
|------|-------------|-------------------|
| Shaheer Zaman | 22I-0805 | Frontend, Integration |
| Najam Hassan | 22I-1332 | ML Model, Backend |
| Aliyan Zafar | 20I-2414 | Data Engineering, Testing |

**Supervisor:** Mr. Almas Khan, FAST NUCES Islamabad

## Institutional Context

Final Year Project (FYP), FAST NUCES Islamabad.

## Future Direction

The long-term vision is to continuously improve translation quality, reduce latency, and expand practical adoption in educational, clinical, and workplace settings to advance inclusive communication infrastructure in Pakistan.

## License

Add an open-source license file if public reuse and contribution are intended.
