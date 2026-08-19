# Render-container font contract

The editorial theme uses the bundled Inter variable font. Any private-beta
render image must install the same TTF that the web preview embeds before it
starts the Python process:

```dockerfile
COPY src/kaivra/assets/fonts/InterVariable.ttf /usr/local/share/fonts/kaivra/
RUN fc-cache -f
```

The file is supplied under the SIL Open Font License 1.1 in
`src/kaivra/assets/fonts/LICENSE.txt`. The matching WOFF2 is embedded in the
self-contained web preview, so both preview and Cairo rendering use Inter when
the render image follows this contract.
