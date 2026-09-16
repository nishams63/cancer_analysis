"use client";
import React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { factors, roc, trend } from "@/lib/demo-data";

// High-contrast neon clinical palette
export const CLINICAL_PALETTE = {
  ctdna: "#00F0FF",      // Electric Cyan (Primary genomic marker)
  cea: "#FF3366",        // Luminous Neon Rose (Protein tumor marker)
  tumorVolume: "#FFB800",// Radiant Amber Gold (Radiographic burden)
  ml: "#FFB800",         // Warm Amber
  dl: "#00F0FF",         // High-tech Cyan
  nlp: "#A855F7",        // Luminous Purple/Violet
  grid: "rgba(255, 255, 255, 0.09)",
  axisText: "#94a3b8",
};

export function RiskRing({
  value,
  tone = "#FF3366",
  label = "High risk",
}: {
  value: number;
  tone?: string;
  label?: string;
}) {
  const data = [{ value }, { value: Math.max(0, 100 - value) }];
  return (
    <div className="risk-ring" role="img" aria-label={`${value} percent ${label}`}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            startAngle={90}
            endAngle={-270}
            innerRadius="74%"
            outerRadius="94%"
            stroke="none"
          >
            <Cell fill={tone} style={{ filter: `drop-shadow(0 0 6px ${tone}66)` }} />
            <Cell fill="rgba(255, 255, 255, 0.07)" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div>
        <strong style={{ color: tone, textShadow: `0 0 12px ${tone}44` }}>{value}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

// Rich custom dark glassmorphism tooltip
function ClinicalTrendTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;
  const units: Record<string, string> = {
    ctDNA: "ng/mL",
    CEA: "µg/L",
    "Tumor volume": "cm³",
    TumorVolume: "cm³",
  };
  return (
    <div
      style={{
        background: "rgba(4, 15, 26, 0.96)",
        border: "1px solid rgba(0, 240, 255, 0.4)",
        borderRadius: "9px",
        padding: "9px 13px",
        boxShadow: "0 8px 30px rgba(0,0,0,0.8), 0 0 15px rgba(0, 240, 255, 0.15)",
        backdropFilter: "blur(10px)",
        minWidth: "155px",
        zIndex: 50,
      }}
    >
      <div
        style={{
          fontWeight: 700,
          color: "#f1f5f9",
          fontSize: "12px",
          marginBottom: "6px",
          borderBottom: "1px solid rgba(255,255,255,0.12)",
          paddingBottom: "4px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>{label} 2026</span>
        <span style={{ fontSize: "10px", color: "#64748b", fontWeight: 500 }}>Cycle Data</span>
      </div>
      {payload.map((entry: any) => {
        const name = entry.name === "TumorVolume" ? "Tumor Volume" : entry.name;
        const unit = units[entry.name] || units[name] || "";
        return (
          <div
            key={entry.name}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "12px",
              margin: "4px 0",
              fontSize: "11px",
            }}
          >
            <span
              style={{
                color: entry.color,
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                fontWeight: 600,
              }}
            >
              <span
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: entry.color,
                  boxShadow: `0 0 6px ${entry.color}`,
                  display: "inline-block",
                }}
              />
              {name}
            </span>
            <span
              style={{
                color: "#ffffff",
                fontWeight: 700,
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            >
              {entry.value}{" "}
              <small style={{ color: "#94a3b8", fontWeight: 400, fontSize: "10px" }}>{unit}</small>
            </span>
          </div>
        );
      })}
    </div>
  );
}

const axisTickStyle = { fill: "#94a3b8", fontSize: 11, fontWeight: 500 };

export function FactorBars() {
  return (
    <div
      className="chart-wrap"
      role="img"
      aria-label="Top toxicity contributors led by creatinine at 24 percent"
      style={{ width: "100%", height: 230 }}
    >
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={factors} layout="vertical" margin={{ left: 10, right: 25, top: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="factorGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#00F0FF" stopOpacity={0.85} />
              <stop offset="100%" stopColor="#0284C7" stopOpacity={0.95} />
            </linearGradient>
          </defs>
          <XAxis type="number" domain={[0, 30]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={90}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#cbd5e1", fontSize: 11, fontWeight: 600 }}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(4, 15, 26, 0.95)",
              border: "1px solid rgba(0, 240, 255, 0.4)",
              borderRadius: "8px",
              color: "#f8fafc",
              fontSize: "12px",
            }}
            formatter={(v) => [`${v}% impact`, "Factor Weight"]}
          />
          <Bar
            dataKey="value"
            fill="url(#factorGrad)"
            radius={[0, 6, 6, 0]}
            barSize={16}
            style={{ filter: "drop-shadow(0 0 5px rgba(0, 240, 255, 0.3))" }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrendChart({
  compact = false,
  months = 6,
  height,
}: {
  compact?: boolean;
  months?: number;
  height?: number;
}) {
  const chartHeight = height || (compact ? 185 : 270);
  const data = trend.slice(-months);

  return (
    <div
      className="chart-wrap"
      role="img"
      aria-label={`Synthetic ctDNA, CEA and tumor volume across ${months} months`}
      style={{ width: "100%", height: chartHeight, position: "relative" }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 12, right: 14, left: -14, bottom: 2 }}>
          <defs>
            {/* Luminous glow gradient for ctDNA primary marker */}
            <linearGradient id="ctdnaGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00F0FF" stopOpacity={0.42} />
              <stop offset="60%" stopColor="#00F0FF" stopOpacity={0.12} />
              <stop offset="100%" stopColor="#00F0FF" stopOpacity={0.01} />
            </linearGradient>
            {/* Subtle luminous gradient for CEA */}
            <linearGradient id="ceaGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#FF3366" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#FF3366" stopOpacity={0.0} />
            </linearGradient>
          </defs>

          {/* High-visibility contrast grid */}
          <CartesianGrid stroke="rgba(255, 255, 255, 0.09)" strokeDasharray="3 3" vertical={false} />

          <XAxis
            dataKey="m"
            axisLine={{ stroke: "rgba(255, 255, 255, 0.15)" }}
            tickLine={false}
            tick={axisTickStyle}
            dy={4}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={axisTickStyle}
            width={34}
            domain={[0, 90]}
          />

          <Tooltip content={<ClinicalTrendTooltip />} />

          {/* 1. Primary ctDNA Area & Curve (High-Contrast Electric Cyan with Glow) */}
          <Area
            type="monotone"
            dataKey="ctDNA"
            name="ctDNA"
            stroke="#00F0FF"
            fill="url(#ctdnaGlow)"
            strokeWidth={2.6}
            dot={{ r: 4, fill: "#00F0FF", stroke: "#031525", strokeWidth: 2 }}
            activeDot={{ r: 6.5, fill: "#00F0FF", stroke: "#ffffff", strokeWidth: 2.5 }}
            style={{ filter: "drop-shadow(0 0 6px rgba(0, 240, 255, 0.5))" }}
          />

          {/* 2. CEA Protein Marker (High-Contrast Vivid Neon Rose) */}
          <Line
            type="monotone"
            dataKey="CEA"
            name="CEA"
            stroke="#FF3366"
            strokeWidth={2.4}
            dot={{ r: 3.5, fill: "#FF3366", stroke: "#031525", strokeWidth: 2 }}
            activeDot={{ r: 6, fill: "#FF3366", stroke: "#ffffff", strokeWidth: 2 }}
            style={{ filter: "drop-shadow(0 0 5px rgba(255, 51, 102, 0.45))" }}
          />

          {/* 3. Tumor Volume (Radiant Amber Gold) */}
          <Line
            type="monotone"
            dataKey="TumorVolume"
            name="Tumor volume"
            stroke="#FFB800"
            strokeWidth={2.3}
            strokeDasharray="4 3"
            dot={{ r: 3.5, fill: "#FFB800", stroke: "#031525", strokeWidth: 2 }}
            activeDot={{ r: 6, fill: "#FFB800", stroke: "#ffffff", strokeWidth: 2 }}
            style={{ filter: "drop-shadow(0 0 5px rgba(255, 184, 0, 0.4))" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RocChart() {
  return (
    <div
      className="chart-wrap"
      role="img"
      aria-label="ROC curves for ML, DL and NLP models"
      style={{ width: "100%", height: 260 }}
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={roc} margin={{ top: 10, right: 16, left: -10, bottom: 4 }}>
          <CartesianGrid stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={axisTickStyle} />
          <YAxis tick={axisTickStyle} width={32} domain={[0, 1]} />
          <Tooltip
            contentStyle={{
              background: "rgba(4, 15, 26, 0.96)",
              border: "1px solid rgba(0, 240, 255, 0.35)",
              borderRadius: "8px",
              color: "#ffffff",
              fontSize: "12px",
            }}
          />
          <Line
            type="monotone"
            dataKey="ML"
            name="Stage 1: ML Toxicity"
            stroke="#FFB800"
            dot={{ r: 3, fill: "#FFB800" }}
            strokeWidth={2.5}
            style={{ filter: "drop-shadow(0 0 4px rgba(255, 184, 0, 0.4))" }}
          />
          <Line
            type="monotone"
            dataKey="DL"
            name="Stage 2: DL Progression"
            stroke="#00F0FF"
            dot={{ r: 3, fill: "#00F0FF" }}
            strokeWidth={2.5}
            style={{ filter: "drop-shadow(0 0 4px rgba(0, 240, 255, 0.5))" }}
          />
          <Line
            type="monotone"
            dataKey="NLP"
            name="Stage 3: NLP Urgency"
            stroke="#A855F7"
            dot={{ r: 3, fill: "#A855F7" }}
            strokeWidth={2.5}
            style={{ filter: "drop-shadow(0 0 4px rgba(168, 85, 247, 0.5))" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
