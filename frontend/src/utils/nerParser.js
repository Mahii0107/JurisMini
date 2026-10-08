import { ENTITY_COLORS } from "../data/mockData";

const REGEX_RULES = [
  { regex: /(?:Supreme Court of India|High Court of Judicature at [A-Za-z]+|High Court of [A-Za-z]+(?:\s+at\s+[A-Za-z]+)?|Sessions Court|Sessions Judge,?\s+[A-Za-z]+)/gi, label: "COURT" },
  { regex: /(?:Chief Justice|CJI|Justice|Hon'ble Mr\. Justice|Hon'ble Ms\. Justice|Hon'ble Dr\. Justice|Hon'ble|Magistrate)\s+[A-Z][A-Za-z\.]*(?:\s+[A-Z][A-Za-z\.]*)*(?:,\s*(?:JJ|J)\.)?/gi, label: "JUDGE" },
  { regex: /(?:Criminal Appeal|Civil Appeal|Special Leave Petition|Writ Petition(?:\s+\(Civil\))?|Criminal Revision(?:\s+Application|\s+Petition)?)\s+No\.\s*\d+\s+of\s+\d{4}/gi, label: "CASE_NUMBER" },
  { regex: /(?:Negotiable Instruments Act,?\s*1881|Indian Penal Code,?\s*1860|Code of Criminal Procedure,?\s*1973|Constitution of India|Code of Civil Procedure,?\s*1908|Evidence Act,?\s*1872|Aadhaar Act|Companies Act,?\s*2013)/gi, label: "STATUTE" },
  { regex: /(?:Section\s+\d+[A-Z]?(?:\(\d+\))?|Article\s+\d+[A-Z]?(?:\(\d+\))?|Rule\s+\d+|Order\s+[IVXLCDM]+|Section\s+138|Article\s+21|Article\s+136|Section\s+302|Section\s+420|Section\s+406|Section\s+161|Section\s+142|Part III)/gi, label: "PROVISION" },
  { regex: /(?:State of [A-Za-z]+|Union of India|Reserve Bank of India|Central Bureau of Investigation|CBI|High Court|Supreme Court)/gi, label: "ORG" },
  { regex: /(?:PW-\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|witness\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/gi, label: "WITNESS" },
  { regex: /(?:Senior Counsel|Senior Advocate|Advocate|Solicitor General|Public Prosecutor|counsel)\s+([A-Z][A-Za-z\.]*(?:\s+[A-Z][A-Za-z\.]*)*)/gi, label: "LAWYER" },
  { regex: /\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{1,2}[\./-]\d{1,2}[\./-]\d{4})\b/gi, label: "DATE" },
  { regex: /([A-Z][A-Za-z\.]*(?:\s+[A-Z][A-Za-z\.]*)*)\s+v\.\s+([A-Z][A-Za-z\.]*(?:\s+[A-Z][A-Za-z\.]*)*(?:,\s*\(\d{4}\)\s*\d+\s*[A-Z]+\s*\d+)?)/gi, label: "PRECEDENT" }
];

export function extractSpansLocal(text) {
  if (!text) return [];
  const spans = [];

  for (const { regex, label } of REGEX_RULES) {
    let match;
    regex.lastIndex = 0;
    while ((match = regex.exec(text)) !== null) {
      let s = match.index;
      let e = match.index + match[0].length;
      while (s < e && /\s/.test(text[s])) s++;
      while (e > s && /[\s\.,;:]/.test(text[e - 1])) e--;
      if (s < e) {
        spans.push({ start: s, end: e, label, text: text.slice(s, e) });
      }
    }
  }

  // Parties detection heuristics
  const petMatch = /(?:appellant|petitioner)\s+([A-Z][A-Za-z\s@]+?)\s+(?:challenged|filed|approached|appealed)/i.exec(text);
  if (petMatch) {
    const s = petMatch.index + petMatch[0].indexOf(petMatch[1]);
    const e = s + petMatch[1].length;
    spans.push({ start: s, end: e, label: "PETITIONER", text: text.slice(s, e) });
  }

  // Sort and remove overlapping spans
  spans.sort((a, b) => (b.end - b.start) - (a.end - a.start));
  const nonOverlapping = [];
  for (const span of spans) {
    const overlaps = nonOverlapping.some(
      ex => (span.start < ex.end && span.end > ex.start)
    );
    if (!overlaps) {
      nonOverlapping.push(span);
    }
  }

  nonOverlapping.sort((a, b) => a.start - b.start);
  return nonOverlapping;
}

export function synthesizeMetadata(text, spans) {
  const groups = {};
  for (const s of spans) {
    if (!groups[s.label]) groups[s.label] = [];
    if (!groups[s.label].includes(s.text.trim())) {
      groups[s.label].push(s.text.trim());
    }
  }

  const court = groups["COURT"]?.[0] || "Indian Court of Law";
  const caseNo = groups["CASE_NUMBER"]?.[0] || "Judicial Case Record";
  const pets = groups["PETITIONER"] || [];
  const resps = groups["RESPONDENT"] || [];

  let title = "Court Judgment Analysis";
  if (pets.length && resps.length) {
    title = `${pets[0]} v. ${resps[0]}`;
  } else if (pets.length) {
    title = `In re: ${pets[0]}`;
  } else if (caseNo !== "Judicial Case Record") {
    title = caseNo;
  }

  let verdict = "Adjudicated Judicial Order";
  const lower = text.toLowerCase();
  if (lower.includes("acquitted") || lower.includes("appeal allowed")) {
    verdict = "Verdict: Acquitted / Appeal Allowed";
  } else if (lower.includes("dismissed")) {
    verdict = "Verdict: Appeal Dismissed";
  } else if (lower.includes("quashed")) {
    verdict = "Verdict: Proceedings Quashed";
  } else if (lower.includes("unanimously held")) {
    verdict = "Verdict: Constitutional Declaration Held";
  }

  return {
    court,
    caseNumber: caseNo,
    title,
    verdict,
    oneSentence: `The ${court} reviewed the legal challenge under ${groups["STATUTE"]?.[0] || "relevant Indian statutes"} and delivered its final judgment.`,
    judges: (groups["JUDGE"] || []).slice(0, 3).join(", ") || "Active Judicial Bench",
    petitioner: pets.slice(0, 2).join(", ") || "Appellant / Petitioner",
    respondent: resps.slice(0, 2).join(", ") || "Respondent / State",
    statutes: (groups["STATUTE"] || []).slice(0, 3).join(" · ") || "Statutory Law",
    provisions: (groups["PROVISION"] || []).slice(0, 4).join(" · ") || "Procedural Provisions",
    lawyers: (groups["LAWYER"] || []).slice(0, 3).join(", ") || "Counsel on Record",
    totalMentions: spans.length
  };
}
