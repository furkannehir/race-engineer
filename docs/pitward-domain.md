# Connecting pitward.com

The website is live at https://pitward.com and https://www.pitward.com.
Both custom domains are active with valid HTTPS certificates on the same Sites
project. All seven authoritative DNS records were verified, and both domains
returned the published Pitward page after refreshing their validation in Sites.

Current nameservers are `launch1.spaceship.net` and `launch2.spaceship.net`.
Add the following records in the domain's Spaceship DNS manager. Host names in
this table are relative to `pitward.com`; use the full name if your provider's
form requires it. The verification values below are public DNS records, not API
credentials. Keep existing mail and unrelated DNS records.

| Type | Host | Value |
| --- | --- | --- |
| CNAME | `www` | `custom-domains.chatgpt.site` |
| TXT | `_openai-site-verification.www` | `openai-site-verification=qunkN-0liWmw61l09tbPYFZ9TYMooUc-XgGDs6xi9dI` |
| TXT | `_cf-custom-hostname.www` | `46dbe81f-dfe7-43e8-9ea6-d7ef559aeb97` |
| A | `@` | `162.159.143.30` |
| A | `@` | `172.66.3.26` |
| TXT | `_openai-site-verification` | `openai-site-verification=m5qi0r_N0rEbbsR5mNW90mLh40DhIkLjIMgHiE84ltM` |
| TXT | `_cf-custom-hostname` | `030fd4f4-b78e-4282-8b9a-d7b9bfaac173` |

After saving the records, refresh both custom domains in Sites and wait for
domain validation and SSL provisioning to become active. No nameserver change
is required. If records conflict at `www` or the root, reconcile them before
adding the web records. This session did not change the DNS provider's records.

The download buttons remain disabled until the approved GitHub `v0.1.0`
release is published with its installer, ZIP, and checksum file.
