# MetaHarness for Nimbalyst

Connect the Nimbalyst run inspector to a locally installed MetaHarness service.
The extension reads run state and artifacts, and sends explicit user actions
through MetaHarness API v1.

## Requirements
- Nimbalyst Extension Dev Tools
- MetaHarness installed locally
- MetaHarness config file
- MetaHarness API v1 support

## Install

From this directory, build the extension bundle:

```sh
npm install
npm run build
```

Load this extension directory in Nimbalyst Extension Dev Tools. The installable
bundle contains `manifest.json`, `dist/index.js`, `dist/index.css`, and
`dist/backend.js`. Enable the MetaHarness backend utility process when prompted
by Nimbalyst.

## Configure

Open the project’s **Settings → Extensions → MetaHarness** page. Set the path to
the local MetaHarness executable and TOML config file, then choose the local
API port (default `8765`). Save the settings and use **Test connection** or
**Run doctor** to check the local service. Enable **Start automatically** if
the extension should start MetaHarness when the project opens. Create and
inspect runs from the MetaHarness panel.

The extension stores these settings in Nimbalyst project ExtensionStorage. It
does not copy the MetaHarness configuration or persist API keys.

## Security model
- localhost only
- control token file
- Nimbalyst backend utility process
- no API keys stored by extension

The backend accepts only loopback HTTP URLs, reads the control token from the
token file created by the local MetaHarness service, and sends it only with
mutating API requests. Read-only requests do not include the token. The
extension does not invoke a shell for API operations or expose provider API
keys to its frontend.
