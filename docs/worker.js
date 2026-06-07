/**
 * Cloudflare Worker CORS Proxy for Yahoo Finance
 * 
 * Deployment Instructions:
 * 1. Go to dash.cloudflare.com and sign up for a free account.
 * 2. Click "Workers & Pages" -> "Create Application" -> "Create Worker".
 * 3. Name it "yfinance-proxy" and click "Deploy".
 * 4. Click "Edit code", paste this entire file in, and click "Save and Deploy".
 * 5. Copy the worker URL (e.g., https://yfinance-proxy.YOUR-ACCOUNT.workers.dev)
 *    and paste it into the proxy_url variable in your app.js file!
 */

export default {
    async fetch(request, env, ctx) {
      // Handle CORS preflight requests
      if (request.method === "OPTIONS") {
        return new Response(null, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400",
          },
        });
      }
  
      const url = new URL(request.url);
      const targetUrl = url.searchParams.get("url");
  
      if (!targetUrl) {
        return new Response("Missing 'url' query parameter", { status: 400 });
      }
  
      try {
        // Forward the request to Yahoo Finance, mocking a standard User-Agent
        const modifiedRequest = new Request(targetUrl, {
          method: request.method,
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "application/json",
          },
          redirect: "follow",
        });
  
        const response = await fetch(modifiedRequest);
  
        // Recreate the response to append CORS headers
        const newResponse = new Response(response.body, response);
        newResponse.headers.set("Access-Control-Allow-Origin", "*");
        
        return newResponse;
  
      } catch (err) {
        return new Response("Proxy Error: " + err.message, { status: 500 });
      }
    },
  };
