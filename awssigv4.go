package llm

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"
)

// unsignableHeaders are never folded into the SigV4 signature (hop-by-hop /
// runtime-owned). Content-Type is deliberately NOT here — AWS requires it signed
// when present.
var unsignableHeaders = map[string]bool{
	"authorization": true, "cache-control": true, "connection": true,
	"content-length": true, "expect": true, "from": true, "keep-alive": true,
	"max-forwards": true, "pragma": true, "proxy-authenticate": true,
	"proxy-authorization": true, "referer": true, "te": true, "trailer": true,
	"transfer-encoding": true, "upgrade": true, "user-agent": true, "x-amzn-trace-id": true,
}

type sigV4Params struct {
	Method          string
	Service         string
	Region          string
	Host            string
	Path            string
	CanonicalQuery  string
	Payload         []byte
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	Signed          map[string]string
	Now             time.Time
}

// signAWSv4 hand-rolls AWS Signature Version 4 on the standard library. Returns
// the headers to attach: Host, X-Amz-Date, X-Amz-Content-Sha256, Authorization,
// and (optionally) X-Amz-Security-Token.
func signAWSv4(p sigV4Params) map[string]string {
	ts := p.Now.UTC()
	amzDate := ts.Format("20060102T150405Z")
	dateStamp := ts.Format("20060102")

	hashedPayload := hexSHA256(p.Payload)

	canonical := map[string]string{}
	for k, v := range p.Signed {
		canonical[strings.ToLower(k)] = collapseWS(v)
	}
	canonical["host"] = collapseWS(p.Host)
	canonical["x-amz-date"] = amzDate
	canonical["x-amz-content-sha256"] = hashedPayload
	if p.SessionToken != "" {
		canonical["x-amz-security-token"] = p.SessionToken
	}
	for h := range canonical {
		if unsignableHeaders[h] {
			delete(canonical, h)
		}
	}

	names := make([]string, 0, len(canonical))
	for k := range canonical {
		names = append(names, k)
	}
	sort.Strings(names)

	var canonicalHeaders strings.Builder
	for _, n := range names {
		canonicalHeaders.WriteString(n)
		canonicalHeaders.WriteByte(':')
		canonicalHeaders.WriteString(canonical[n])
		canonicalHeaders.WriteByte('\n')
	}
	signedHeaders := strings.Join(names, ";")

	canonicalRequest := p.Method + "\n" + p.Path + "\n" + p.CanonicalQuery + "\n" +
		canonicalHeaders.String() + "\n" + signedHeaders + "\n" + hashedPayload

	credentialScope := dateStamp + "/" + p.Region + "/" + p.Service + "/aws4_request"
	stringToSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + credentialScope + "\n" +
		hexSHA256([]byte(canonicalRequest))

	kDate := hmacSHA256([]byte("AWS4"+p.SecretAccessKey), dateStamp)
	kRegion := hmacSHA256(kDate, p.Region)
	kService := hmacSHA256(kRegion, p.Service)
	kSigning := hmacSHA256(kService, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(kSigning, stringToSign))

	authorization := "AWS4-HMAC-SHA256 Credential=" + p.AccessKeyID + "/" + credentialScope +
		", SignedHeaders=" + signedHeaders + ", Signature=" + signature

	out := map[string]string{
		"Host":                 p.Host,
		"X-Amz-Date":           amzDate,
		"X-Amz-Content-Sha256": hashedPayload,
		"Authorization":        authorization,
	}
	if p.SessionToken != "" {
		out["X-Amz-Security-Token"] = p.SessionToken
	}
	return out
}

func buildCanonicalQuery(query map[string]string) string {
	if len(query) == 0 {
		return ""
	}
	keys := make([]string, 0, len(query))
	for k := range query {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, rfc3986Encode(k)+"="+rfc3986Encode(query[k]))
	}
	return strings.Join(parts, "&")
}

func rfc3986Encode(s string) string {
	e := url.QueryEscape(s)
	e = strings.ReplaceAll(e, "+", "%20")
	e = strings.ReplaceAll(e, "%7E", "~")
	return e
}

func hexSHA256(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

var wsRe = regexp.MustCompile(`\s+`)

func collapseWS(s string) string {
	return wsRe.ReplaceAllString(strings.TrimSpace(s), " ")
}
