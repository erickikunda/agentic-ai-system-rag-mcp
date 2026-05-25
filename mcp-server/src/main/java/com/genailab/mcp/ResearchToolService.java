package com.genailab.mcp;

import java.util.Arrays;
import java.util.List;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

@Component
public class ResearchToolService {

    @Tool(
            name = "normalize_claim",
            description = "Normalize a research claim into a concise, citation-ready statement and extract key terms.")
    public ClaimNormalizationResult normalizeClaim(
            @ToolParam(description = "Raw claim from a user, paper, note, or retrieved passage.", required = true)
            String claim) {
        String normalized = claim.strip().replaceAll("\\s+", " ");
        if (!normalized.endsWith(".")) {
            normalized = normalized + ".";
        }
        List<String> keyTerms = Arrays.stream(normalized.split("[^A-Za-z0-9_-]+"))
                .filter(term -> term.length() > 4)
                .map(String::toLowerCase)
                .distinct()
                .limit(8)
                .toList();
        return new ClaimNormalizationResult(
                normalized,
                keyTerms,
                "Heuristic normalization only; verify against retrieved citations before using in an answer.");
    }

    @Tool(
            name = "lookup_venue_metadata",
            description = "Return lightweight venue metadata and review cautions for research-source provenance checks.")
    public VenueMetadata lookupVenueMetadata(
            @ToolParam(description = "Venue or source name, for example arXiv, NeurIPS, RFC, or internal design note.", required = true)
            String venue) {
        String normalized = venue.strip().toLowerCase();
        if (normalized.contains("arxiv")) {
            return new VenueMetadata(venue, "preprint repository", "not peer reviewed by default",
                    "Treat as useful but not final; prefer corroborating sources.");
        }
        if (normalized.contains("neurips") || normalized.contains("icml") || normalized.contains("acl")) {
            return new VenueMetadata(venue, "conference", "peer reviewed",
                    "Check paper year, workshop/main-track status, and follow-up corrections.");
        }
        if (normalized.contains("rfc")) {
            return new VenueMetadata(venue, "standards document", "community review",
                    "Distinguish normative language from explanatory examples.");
        }
        return new VenueMetadata(venue, "unknown or local source", "unknown",
                "Require stronger citation metadata before treating as authoritative.");
    }

    @Tool(
            name = "build_reading_plan",
            description = "Create a short reading plan for studying a GenAI engineering topic.")
    public ReadingPlan buildReadingPlan(
            @ToolParam(description = "Topic the user wants to study.", required = true)
            String topic,
            @ToolParam(description = "Current skill level or context.", required = false)
            String level) {
        String audience = (level == null || level.isBlank()) ? "senior CS student" : level.strip();
        return new ReadingPlan(topic, List.of(
                "Skim the highest-level architecture notes and write down the system boundary.",
                "Read implementation details with attention to failure modes and operational trade-offs.",
                "Run or inspect one concrete example, then explain what state changes after each step.",
                "Write three production questions: latency, correctness, and rollback or recovery."),
                "A " + audience + " should be able to explain the concept, its failure modes, and when not to use it.");
    }
}
