#!/usr/bin/env node

import { createSign } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const cachePath = process.env.GITHUB_APP_TOKEN_CACHE;

if (cachePath) {
  try {
    const cached = JSON.parse(await readFile(cachePath, "utf8"));
    if (Date.parse(cached.expiresAt) > Date.now() + 5 * 60 * 1000) {
      process.stderr.write(`::add-mask::${cached.token}\n`);
      process.stdout.write(cached.token);
      process.exit(0);
    }
  } catch {
    // A missing, stale, or invalid cache is replaced below.
  }
}

const appId = process.env.GITHUB_APP_ID;
const installationId = process.env.GITHUB_APP_INSTALLATION_ID;
const privateKeyPath = process.env.GITHUB_APP_PRIVATE_KEY_FILE;
const repository = process.env.GITHUB_APP_REPOSITORY;

if (!appId || !installationId || !privateKeyPath) {
  throw new Error("GitHub App ID, installation ID, and private key file are required");
}

const now = Math.floor(Date.now() / 1000);
const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
const unsignedToken = `${encode({ alg: "RS256", typ: "JWT" })}.${encode({
  iat: now - 60,
  exp: now + 9 * 60,
  iss: appId,
})}`;
const privateKey = (await readFile(privateKeyPath, "utf8")).replaceAll("\\n", "\n");
const signer = createSign("RSA-SHA256");
signer.update(unsignedToken);
const jwt = `${unsignedToken}.${signer.sign(privateKey, "base64url")}`;

const response = await fetch(
  `https://api.github.com/app/installations/${installationId}/access_tokens`,
  {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${jwt}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "kernel-tinker-rl-weekly-maintenance",
    },
    body: repository ? JSON.stringify({ repositories: [repository] }) : undefined,
  },
);

if (!response.ok) {
  throw new Error(`GitHub App token request failed with HTTP ${response.status}`);
}

const result = await response.json();
if (!result.token || !result.expires_at) {
  throw new Error("GitHub App token response was incomplete");
}

process.stderr.write(`::add-mask::${result.token}\n`);
if (cachePath) {
  await writeFile(
    cachePath,
    JSON.stringify({ token: result.token, expiresAt: result.expires_at }),
    { mode: 0o600 },
  );
}
process.stdout.write(result.token);
