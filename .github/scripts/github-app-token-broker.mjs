#!/usr/bin/env node

// Serves installation tokens for one repository with fixed permissions, so the GitHub App private
// key stays in this process and never reaches the agent. The key is read from stdin rather than
// the environment, so it does not show up in /proc/<pid>/environ.

import { createSign } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

const rawPrivateKey = readFileSync(0, "utf8");
const {
  GITHUB_APP_ID: appId,
  GITHUB_APP_INSTALLATION_ID: installationId,
  GITHUB_APP_REPOSITORY: repository,
  GITHUB_APP_PERMISSIONS: permissionsJson,
  GITHUB_APP_BROKER_HOST: host,
  GITHUB_APP_BROKER_PORT: port,
} = process.env;

if (!appId || !installationId || !rawPrivateKey || !repository || !permissionsJson || !host || !port) {
  throw new Error(
    "GitHub App ID, installation ID, private key, repository, permissions, and broker address are required",
  );
}

const privateKey = rawPrivateKey.replaceAll("\\n", "\n");
const permissions = JSON.parse(permissionsJson);
let cached;
let pending;

async function mint() {
  const now = Math.floor(Date.now() / 1000);
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
  const unsignedToken = `${encode({ alg: "RS256", typ: "JWT" })}.${encode({
    iat: now - 60,
    exp: now + 9 * 60,
    iss: appId,
  })}`;
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
        "User-Agent": "github-app-token-broker",
      },
      body: JSON.stringify({ repositories: [repository], permissions }),
    },
  );
  if (!response.ok) {
    throw new Error(`GitHub App token request failed with HTTP ${response.status}`);
  }
  const result = await response.json();
  if (!result.token || !result.expires_at) {
    throw new Error("GitHub App token response was incomplete");
  }
  console.log(
    `minted a token for ${repository} expiring ${result.expires_at} with ${JSON.stringify(result.permissions)}`,
  );
  return { token: result.token, expiresAt: result.expires_at };
}

async function token() {
  if (cached && Date.parse(cached.expiresAt) > Date.now() + 5 * 60 * 1000) {
    return cached.token;
  }
  pending ??= mint()
    .then((minted) => {
      cached = minted;
      return minted.token;
    })
    .finally(() => {
      pending = undefined;
    });
  return pending;
}

createServer(async (request, response) => {
  if (request.method !== "GET" || request.url !== "/token") {
    response.writeHead(404).end();
    return;
  }
  try {
    response.writeHead(200, { "Content-Type": "text/plain" }).end(await token());
  } catch (error) {
    console.error(error.message);
    response.writeHead(502).end();
  }
}).listen(Number(port), host, () => {
  console.log(`serving ${repository} tokens on ${host}:${port}`);
});
