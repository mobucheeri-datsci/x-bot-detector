# Privacy Policy for X Bot Detector

**Effective date:** 8 May 2026
**Last updated:** 8 May 2026

This Privacy Policy describes how the X Bot Detector Chrome extension (the "extension") handles information when you use it. By installing and using the extension, you acknowledge this policy.

## 1. Who is responsible

The extension is developed and maintained as an open-source project. The contact channel for any questions about this policy or about data handling is the project's GitHub Issues page:

https://github.com/mobucheeri-datsci/x-bot-detector/issues

The extension is a non-commercial open-source project. The project author acts as the data controller for any processing within the meaning of GDPR Article 4.

## 2. What data the extension processes

### Public profile information of X accounts you visit

When you visit an X profile or thread page, the extension reads publicly visible information from the page, including:

- Account username (handle), display name, and bio text
- Follower count, following count, tweet count, listed count
- Account creation date (account age in days)
- Verified status
- Default avatar status
- Reply usernames and verified status on thread pages

This information is already publicly visible to anyone visiting the profile or thread page. The extension does not access data that requires authentication or special permissions, does not access private messages, and does not access protected accounts you do not have permission to view.

### No data about you, the extension user

The extension does not collect or transmit:

- Your name, email, phone number, address, age, or identification numbers
- Your X account credentials, login information, session cookies, or DM contents
- Your browsing history outside the current X.com or twitter.com page
- Your activity on other websites or other Chrome tabs
- Mouse position, keystroke logs, scroll positions, or other behavioural data

## 3. How the data is used

Profile features extracted from X pages are sent over HTTPS to a backend service hosted on Hugging Face Spaces at:

https://mobucheeri-twitter-bot-detector.hf.space

The backend runs an XGBoost machine learning model that returns a bot likelihood score (0 to 100), a label (HUMAN, UNCERTAIN, or BOT), and a breakdown of which features influenced the score. The response is displayed in the extension's user interface on the X page.

The application code on the backend does not log or persist request data, response data, or IP addresses beyond the in-memory request/response cycle. The backend source code is publicly auditable in the project repository.

Note that Hugging Face, as the platform host, may collect standard HTTP infrastructure data (such as transient IP addresses for routing) per their own privacy policy at https://huggingface.co/privacy. The extension developer does not control or have access to Hugging Face's infrastructure logs.

## 4. Local storage

The extension uses `chrome.storage.local` to cache the most recent score result so that the toolbar popup can redisplay it without making another network request. This cache:

- Stores only the most recent score result (a single record)
- Is overwritten each time a new profile or thread is analysed
- Lives in your browser, on your device, and is never transmitted off your device
- Is removed when you uninstall the extension or clear Chrome extension storage

The cache contains the public profile features and the model's response for the most recent profile scored. It contains no personal data about you, the extension user.

## 5. Cookies and similar technologies

The extension does not use cookies, web beacons, tracking pixels, fingerprinting, local storage on web pages, or any similar tracking technology. The only persistence mechanism the extension uses is Chrome's extension storage API (`chrome.storage.local`), described in section 4, which is local to your browser and never transmitted.

## 6. Data sharing

The extension does not sell, rent, share, transfer, or otherwise disclose any data to third parties. The only external service the extension communicates with is the Hugging Face Space backend described in section 3, which is operated by the same developer and exists solely to run the scoring model.

The extension does not use analytics services, advertising networks, social media tracking, or any third-party SDKs.

## 7. Sub-processors

The extension relies on one sub-processor:

- **Hugging Face, Inc.** (United States). Hugging Face hosts the FastAPI backend on its Spaces platform. The application code on the backend does not log requests; however, Hugging Face's underlying infrastructure may log standard HTTP metadata per their privacy policy: https://huggingface.co/privacy.

No other sub-processors, data processors, or third-party services are involved in the operation of the extension.

## 8. International data transfers

The Hugging Face Space backend is hosted on Hugging Face's infrastructure, which may be located in the United States or other regions. By using the extension, you understand that profile features may be transmitted to and processed in jurisdictions outside your own. The application code does not retain or log this data; transient processing occurs only to compute the score and return it.

For users in the European Economic Area, this transfer is justified on the basis of legitimate interest (analysing publicly available data to compute a bot likelihood score, at the user's explicit initiation by visiting an X page) under GDPR Article 6(1)(f).

## 9. Data retention

| Data | Location | Retention period |
|---|---|---|
| Most recent score result | Your browser (chrome.storage.local) | Until next analysis or extension uninstall |
| API request data | Backend memory | Duration of the HTTP request only; not logged |
| API response data | Backend memory | Duration of the HTTP request only; not logged |
| Hugging Face infrastructure logs | Hugging Face platform | Per Hugging Face's own retention policies, outside the extension developer's control |

## 10. Automated decision-making

The extension uses an automated machine learning model (XGBoost) to compute a bot likelihood score. The score is a probabilistic estimate, not a definitive determination, and is always presented alongside the feature contributions that influenced it so you can understand the result and apply your own judgement.

The score does not produce legal effects or similarly significant effects on the data subject (the X account being scored). It is informational tooling intended to help users contextualise the accounts they encounter, not to take consequential actions on their behalf.

If you are an X account holder and wish to challenge a score the extension produces about your account, you may open an issue using the "Inaccurate Bot Score" template at:

https://github.com/mobucheeri-datsci/x-bot-detector/issues/new/choose

## 11. Special categories of personal data

The extension does not knowingly process any special categories of personal data as defined in GDPR Article 9 (such as data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic or biometric data, health data, or data concerning sexual orientation), nor any sensitive personal information as defined in CCPA / CPRA.

## 12. EU representative

The extension is a non-commercial open-source project that processes a minimal volume of data for a single, limited purpose. On that basis, the project is exempt from the requirement to designate an EU representative under GDPR Article 27(2). For any data protection inquiries from EU users, the contact channel in section 1 applies and is actively monitored.

## 13. What happens when you uninstall

When you uninstall the extension from `chrome://extensions`:

- All locally cached data is removed automatically by Chrome.
- The extension stops running and stops making any network requests from your browser.
- Any data transmitted in previous requests is not retrievable, because it was never persisted on the backend.

If you also wish to clear cached data without uninstalling, you can do so via Chrome's Settings menu under Privacy and security, by clearing extension data.

## 14. Your rights

### Under the GDPR (EEA users)

If you are in the European Economic Area, you have the following rights under the General Data Protection Regulation, to the extent that any data processed by the extension qualifies as your personal data:

- Right of access (Article 15)
- Right to rectification (Article 16)
- Right to erasure, also known as the right to be forgotten (Article 17)
- Right to restriction of processing (Article 18)
- Right to data portability (Article 20)
- Right to object to processing (Article 21)
- Right to lodge a complaint with a supervisory authority (Article 77)

Because the extension does not retain personal data on the backend and the local cache is under your direct control via Chrome, exercising rights of access, correction, or deletion is typically straightforward: clear the local cache by uninstalling the extension or clearing Chrome extension data. For other inquiries, contact us via the GitHub Issues link in section 1.

### Under the CCPA and CPRA (California users)

If you are a California resident, you have the following rights under the California Consumer Privacy Act and California Privacy Rights Act:

- Right to know what personal information is collected
- Right to know whether personal information is sold or disclosed (the extension does not sell or disclose any)
- Right to opt out of the sale of personal information (not applicable; the extension does not sell any)
- Right to access personal information held about you
- Right to deletion
- Right to non-discrimination for exercising any of these rights

To exercise these rights, contact us via the GitHub Issues link in section 1.

### Under other regional privacy laws

Equivalent rights under other applicable privacy laws (such as the UK GDPR, LGPD in Brazil, or PIPEDA in Canada) are honoured to the same extent. Contact us via GitHub Issues to make any request.

## 15. Children's privacy

The extension is not directed at children under the age of 13, and we do not knowingly process information from anyone under 13. Use of X is itself subject to X's age requirement of 13 or older. If a parent or guardian believes a child under 13 has used the extension, contact us via GitHub Issues and we will take appropriate action.

## 16. Security

All communication between the extension and the backend uses HTTPS with TLS encryption. The local cache is protected by Chrome's extension sandboxing model, which restricts access from other extensions and websites.

The extension is open-source and its full source code is available for independent security review at:

https://github.com/mobucheeri-datsci/x-bot-detector

If you discover a security vulnerability, please report it via the GitHub Issues link rather than disclosing it publicly, so that we can address it promptly.

## 17. Changes to this policy

This Privacy Policy may be updated from time to time. Material changes will be reflected in updates to the extension and noted in the project README. The "Last updated" date at the top of this document indicates when the policy was last revised. Continued use of the extension after a revision constitutes acceptance of the updated policy.

## 18. Compliance with the Chrome Web Store Developer Program Policies

The extension complies with Google's Chrome Web Store Developer Program Policies, including the Limited Use requirements for user data. Specifically, the extension:

- Uses extracted profile features only for the single purpose of computing a bot likelihood score for the user
- Does not transfer user data to third parties for serving advertisements
- Does not transfer or use user data to determine creditworthiness or for lending purposes
- Does not allow humans to read collected data, except as required by law, to provide a service the user requested, or for security and abuse investigations

## 19. Contact

For questions about this Privacy Policy, to report a concern, or to exercise any rights described in section 14, open an issue at:

https://github.com/mobucheeri-datsci/x-bot-detector/issues

The repository is actively monitored.