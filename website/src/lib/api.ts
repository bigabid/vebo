// Mock API functions for AWS Athena Explorer
// These would be replaced with actual API calls to your backend

export interface DataSource { name: string }
export interface Catalog { name: string }
export interface Database { name: string }

export interface Table {
  name: string;
}

export interface PartitionData {
  partitionKeys: string[];
  partitions: Record<string, string>[];
}

export interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  stage: string;
  message: string;
  details?: Record<string, any>;
}

export interface InsightsJob {
  jobId: string;
  status: 'running' | 'complete' | 'error' | 'cancelled';
  progress?: number;
  message?: string;
  insights?: InsightsData;
  logs: LogEntry[];
}

export interface InsightsData {
  table: string;
  appliedFilters: Record<string, string[]>;
  rowCount: number;
  partitionSummary: {
    partitionKeys: string[];
    selectedCount: number;
    totalDistinct: number;
  };
  columns: ColumnInsight[];
  candidateKeys?: Array<{
    columns: string[];
    uniqueness: number; // 0..1
    noNulls: boolean;
    confidence?: number; // 0..1
    reason?: string;
  }>;
  primaryKeys?: Array<{
    columns: string[];
    uniqueness: number;
    noNulls: boolean;
    confidence: number;
    reason?: string;
  }>;
  crossColumn?: CrossColumnResult[];
}

export interface CrossColumnResult {
  checkId?: string;
  name?: string;
  status?: string;
  message?: string;
  columns?: (string | undefined)[] | null;
  details?: Record<string, any>;
}

export interface TextPattern {
  regex: string;
  description: string;
  match_count: number;
  match_ratio: number;
  confidence: number;
  examples: string[];
}

export interface TextPatternCheck {
  basic_patterns?: {
    email_like?: { count: number; ratio: number };
    phone_like?: { count: number; ratio: number };
    url_like?: { count: number; ratio: number };
  };
  inferred_patterns?: TextPattern[];
  status?: string;
  message?: string;
}

export interface ColumnCheck {
  check_id: string;
  rule_id: string;
  name: string;
  description?: string;
  status: string;
  score: number;
  message: string;
  details?: any;
  execution_time_ms: number;
  timestamp: string;
}

export interface ColumnInsight {
  name: string;
  type: string;
  nullRatio: number;
  valueType?: 'categorical' | 'continuous';
  topValues?: { value: string; count: number }[];
  numeric?: { min: number; max: number; avg: number; median?: number; std?: number };
  temporal?: { min: string; max: string };
  basic?: {
    uniqueCount?: number;
    uniqueRatio?: number;
    duplicateCount?: number;
    duplicateRatio?: number;
    nullCount?: number;
    nullRatioDetailed?: number;
    mostCommonValue?: string | number | null;
    mostCommonFrequency?: number;
    mostCommonFrequencyRatio?: number;
    mostCommonValueNote?: string;
    isConstantColumn?: boolean;
  };
  checks?: ColumnCheck[];
  textPatterns?: TextPatternCheck;
}

// Mock delay utility
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock data for hierarchy: dataSource -> catalog -> database -> tables
const MOCK_DATASOURCES: string[] = ['AwsDataCatalog', 'MyLakeFormation'];

const MOCK_CATALOGS: Record<string, string[]> = {
  AwsDataCatalog: ['AwsDataCatalog'],
  MyLakeFormation: ['lf-catalog']
};

const MOCK_DATABASES: Record<string, Record<string, string[]>> = {
  AwsDataCatalog: {
    AwsDataCatalog: ['default', 'analytics', 'sales']
  },
  MyLakeFormation: {
    'lf-catalog': ['governed', 'bi']
  }
};

// Keyed by `${dataSource}.${catalog}.${database}`
const MOCK_TABLES_BY_DB: Record<string, Table[]> = {
  'AwsDataCatalog.AwsDataCatalog.default': [
    { name: 'events_prod' },
    { name: 'impressions_daily' }
  ],
  'AwsDataCatalog.AwsDataCatalog.analytics': [
    { name: 'user_sessions' },
    { name: 'product_analytics' }
  ],
  'AwsDataCatalog.AwsDataCatalog.sales': [
    { name: 'purchases_2025' }
  ],
  'MyLakeFormation.lf-catalog.governed': [
    { name: 'events_prod' }
  ],
  'MyLakeFormation.lf-catalog.bi': [
    { name: 'product_analytics' }
  ]
};

const MOCK_PARTITIONS: Record<string, PartitionData> = {
  'events_prod': {
    partitionKeys: ['dt', 'country'],
    partitions: [
      { dt: '2025-08-30', country: 'US' },
      { dt: '2025-08-31', country: 'US' },
      { dt: '2025-09-01', country: 'US' },
      { dt: '2025-08-30', country: 'IL' },
      { dt: '2025-08-31', country: 'IL' },
      { dt: '2025-09-01', country: 'IL' },
      { dt: '2025-08-30', country: 'CA' },
      { dt: '2025-08-31', country: 'CA' }
    ]
  },
  'purchases_2025': {
    partitionKeys: ['year', 'month'],
    partitions: [
      { year: '2025', month: '08' },
      { year: '2025', month: '09' },
      { year: '2025', month: '10' }
    ]
  },
  'impressions_daily': {
    partitionKeys: [],
    partitions: []
  }
};

const MOCK_INSIGHTS: InsightsData = {
  table: 'events_prod',
  appliedFilters: { dt: ['2025-08-31', '2025-09-01'], country: ['US', 'IL'] },
  rowCount: 1234567,
  partitionSummary: {
    partitionKeys: ['dt', 'country'],
    selectedCount: 4,
    totalDistinct: 120
  },
  columns: [
    {
      name: 'user_id',
      type: 'textual',
      nullRatio: 0.01,
      topValues: [
        { value: 'u_123', count: 4321 },
        { value: 'u_456', count: 3987 },
        { value: 'u_789', count: 3654 },
        { value: 'u_012', count: 3201 },
        { value: 'u_345', count: 2876 }
      ],
      textPatterns: {
        basic_patterns: {
          email_like: { count: 0, ratio: 0 },
          phone_like: { count: 0, ratio: 0 },
          url_like: { count: 0, ratio: 0 }
        },
        inferred_patterns: [
          {
            regex: '^u_\\d{3}$',
            description: 'User ID format (u_ prefix + 3 digits)',
            match_count: 1234560,
            match_ratio: 0.999,
            confidence: 95.2,
            examples: ['u_123', 'u_456', 'u_789']
          },
          {
            regex: '^[a-z]_\\d+$',
            description: 'Letter prefix with underscore and digits',
            match_count: 1234560,
            match_ratio: 0.999,
            confidence: 92.8,
            examples: ['u_123', 'u_456']
          }
        ],
        status: 'passed',
        message: 'Text pattern analysis completed. Found 2 inferred patterns.'
      }
    },
    {
      name: 'revenue',
      type: 'double',
      nullRatio: 0.10,
      numeric: { min: 0.0, max: 999.99, avg: 34.52 }
    },
    {
      name: 'event_time',
      type: 'timestamp',
      nullRatio: 0.00,
      temporal: { min: '2025-08-31T00:00:00Z', max: '2025-09-01T23:59:59Z' }
    },
    {
      name: 'category',
      type: 'textual',
      nullRatio: 0.05,
      topValues: [
        { value: 'electronics', count: 125000 },
        { value: 'clothing', count: 98000 },
        { value: 'books', count: 87000 },
        { value: 'home', count: 76000 },
        { value: 'sports', count: 65000 }
      ],
      textPatterns: {
        basic_patterns: {
          email_like: { count: 0, ratio: 0 },
          phone_like: { count: 0, ratio: 0 },
          url_like: { count: 0, ratio: 0 }
        },
        inferred_patterns: [
          {
            regex: '^[a-z]+$',
            description: 'All lowercase letters',
            match_count: 451000,
            match_ratio: 0.98,
            confidence: 94.9,
            examples: ['electronics', 'clothing', 'books']
          },
          {
            regex: '^[a-zA-Z]+$',
            description: 'Single word',
            match_count: 451000,
            match_ratio: 0.98,
            confidence: 94.9,
            examples: ['electronics', 'clothing']
          }
        ],
        status: 'passed',
        message: 'Text pattern analysis completed. Found 2 inferred patterns.'
      }
    },
    {
      name: 'session_duration',
      type: 'integer',
      nullRatio: 0.02,
      numeric: { min: 1, max: 7200, avg: 245 }
    }
  ]
};

// API Functions
export const fetchDataSources = async (): Promise<{ dataSources: string[] }> => {
  try {
    const res = await fetch('/api/data-sources');
    if (!res.ok) throw new Error('Failed');
    return res.json();
  } catch {
    await delay(300);
    return { dataSources: MOCK_DATASOURCES };
  }
};

export const fetchCatalogs = async (
  dataSource: string
): Promise<{ catalogs: string[] }> => {
  try {
    const res = await fetch(`/api/catalogs?dataSource=${encodeURIComponent(dataSource)}`);
    if (!res.ok) throw new Error('Failed');
    return res.json();
  } catch {
    await delay(300);
    return { catalogs: MOCK_CATALOGS[dataSource] || [] };
  }
};

export const fetchDatabases = async (
  dataSource: string,
  catalog: string
): Promise<{ databases: string[] }> => {
  try {
    const res = await fetch(`/api/databases?catalog=${encodeURIComponent(catalog)}`);
    if (!res.ok) throw new Error('Failed');
    return res.json();
  } catch {
    await delay(400);
    return { databases: (MOCK_DATABASES[dataSource]?.[catalog]) || [] };
  }
};

export const fetchTables = async (
  dataSource: string,
  catalog: string,
  database: string
): Promise<{ tables: Table[] }> => {
  try {
    const res = await fetch(`/api/tables?catalog=${encodeURIComponent(catalog)}&database=${encodeURIComponent(database)}`);
    if (!res.ok) throw new Error('Failed');
    return res.json();
  } catch {
    await delay(800);
    const key = `${dataSource}.${catalog}.${database}`;
    return { tables: MOCK_TABLES_BY_DB[key] || [] };
  }
};

export const fetchPartitions = async (
  table: string,
  params?: { catalog?: string; database?: string }
): Promise<PartitionData> => {
  try {
    if (params?.catalog && params?.database) {
      const res = await fetch(`/api/partitions?catalog=${encodeURIComponent(params.catalog)}&database=${encodeURIComponent(params.database)}&table=${encodeURIComponent(table)}`);
      if (!res.ok) throw new Error('Failed');
      return res.json();
    }
    throw new Error('Missing real params');
  } catch {
    await delay(600);
    return MOCK_PARTITIONS[table] || { partitionKeys: [], partitions: [] };
  }
};

const INSIGHTS_API = (import.meta as any).env?.VITE_INSIGHTS_API || 'http://localhost:8000';

export const getJobStatus = async (jobId: string): Promise<InsightsJob> => {
  const res = await fetch(`${INSIGHTS_API}/insights/status?jobId=${encodeURIComponent(jobId)}`);
  if (!res.ok) {
    throw new Error('Failed to get job status');
  }
  return res.json();
};

export const cancelJob = async (jobId: string): Promise<{ status: string; message: string }> => {
  const response = await fetch(`${INSIGHTS_API}/insights/cancel`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ jobId }),
  });
  if (!response.ok) {
    throw new Error('Failed to cancel job');
  }
  return response.json();
};

export const startInsightsJob = async (
  params: {
    dataSource: string;
    catalog: string;
    database: string;
    table: string;
    partitions: Record<string, string[]>;
  }
): Promise<{ jobId: string; status: string }> => {
  // 1) Ask Node server to start Athena query and return executionId
  const res = await fetch('/api/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      catalog: params.catalog,
      database: params.database,
      table: params.table,
      partitions: params.partitions,
    }),
  });
  if (!res.ok) {
    throw new Error('Failed to start Athena query');
  }
  const { executionId } = await res.json();

  // 2) Tell Python service to start insights processing for that executionId
  const pyRes = await fetch(`${INSIGHTS_API}/insights/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      executionId,
      table: params.table,
      catalog: params.catalog,
      database: params.database,
      appliedFilters: params.partitions,
    }),
  });
  if (!pyRes.ok) {
    throw new Error('Failed to start insights job');
  }
  return pyRes.json();
};