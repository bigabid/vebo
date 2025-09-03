import express from 'express';
import cors from 'cors';
import { AthenaClient, ListDataCatalogsCommand, ListDatabasesCommand, ListTableMetadataCommand, StartQueryExecutionCommand } from '@aws-sdk/client-athena';
import { GlueClient, GetPartitionsCommand, GetTableCommand } from '@aws-sdk/client-glue';

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 8787;

const region = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-1';
const athena = new AthenaClient({ region });
const glue = new GlueClient({ region });

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.get('/api/data-sources', async (_req, res) => {
  try {
    const out = await athena.send(new ListDataCatalogsCommand({}));
    const names = (out.DataCatalogsSummary || []).map(d => d?.CatalogName).filter(Boolean);
    if (!names.includes('AwsDataCatalog')) names.unshift('AwsDataCatalog');
    res.json({ dataSources: names });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list data sources' });
  }
});

app.get('/api/catalogs', async (req, res) => {
  try {
    const { dataSource } = req.query;
    if (!dataSource) return res.status(400).json({ error: 'Missing dataSource' });
    // For Athena, the data source corresponds to a single catalog of the same name
    res.json({ catalogs: [String(dataSource)] });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list catalogs' });
  }
});

app.get('/api/databases', async (req, res) => {
  try {
    const { catalog } = req.query;
    if (!catalog) return res.status(400).json({ error: 'Missing catalog' });
    const out = await athena.send(new ListDatabasesCommand({ CatalogName: String(catalog) }));
    const dbs = (out.DatabaseList || []).map(d => d?.Name).filter(Boolean);
    res.json({ databases: dbs });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list databases' });
  }
});

app.get('/api/tables', async (req, res) => {
  try {
    const { catalog, database } = req.query;
    if (!catalog || !database) return res.status(400).json({ error: 'Missing catalog or database' });
    const out = await athena.send(new ListTableMetadataCommand({ CatalogName: String(catalog), DatabaseName: String(database) }));
    const tables = (out.TableMetadataList || []).map(t => ({ name: t?.Name })).filter(t => t.name);
    res.json({ tables });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list tables' });
  }
});

app.get('/api/partitions', async (req, res) => {
  try {
    const { catalog, database, table } = req.query;
    if (!catalog || !database || !table) return res.status(400).json({ error: 'Missing catalog, database or table' });

    // Get partition keys
    const tableResp = await glue.send(new GetTableCommand({
      CatalogId: undefined, // default account
      DatabaseName: String(database),
      Name: String(table),
    }));
    const partitionKeys = (tableResp.Table?.PartitionKeys || []).map(k => k?.Name).filter(Boolean);

    if (!partitionKeys.length) {
      return res.json({ partitionKeys: [], partitions: [] });
    }

    // List partitions (first page)
    const partResp = await glue.send(new GetPartitionsCommand({
      DatabaseName: String(database),
      TableName: String(table),
      MaxResults: 500,
    }));

    const partitions = (partResp.Partitions || []).map(p => {
      const obj = {};
      (p.Values || []).forEach((val, idx) => {
        const key = partitionKeys[idx];
        if (key) obj[key] = val;
      });
      return obj;
    });

    res.json({ partitionKeys, partitions });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list partitions' });
  }
});

// Start an Athena query execution based on user selections
app.post('/api/execute', async (req, res) => {
  try {
    const { catalog, database, table, partitions } = req.body || {};
    if (!catalog || !database || !table) {
      return res.status(400).json({ error: 'Missing catalog, database or table' });
    }

    const outputLocation = process.env.ATHENA_OUTPUT_S3;
    if (!outputLocation) {
      return res.status(500).json({ error: 'ATHENA_OUTPUT_S3 env var is required (e.g., s3://my-bucket/athena-results/)' });
    }

    // Discover partition key types to properly type literals in WHERE clause
    const tableResp = await glue.send(new GetTableCommand({
      DatabaseName: String(database),
      Name: String(table),
    }));
    const partitionKeyTypeByName = {};
    for (const k of (tableResp.Table?.PartitionKeys || [])) {
      if (k?.Name) {
        partitionKeyTypeByName[k.Name] = String(k.Type || '').toLowerCase();
      }
    }

    const escapeSingle = (s) => String(s).replace(/'/g, "''");
    const formatLiteral = (val, type) => {
      const t = (type || '').toLowerCase();
      if (t === 'date') {
        return `DATE '${escapeSingle(val)}'`;
      }
      if (t === 'timestamp') {
        return `TIMESTAMP '${escapeSingle(val)}'`;
      }
      if (
        t === 'int' || t === 'integer' || t === 'bigint' || t === 'smallint' || t === 'tinyint' ||
        t === 'float' || t === 'real' || t === 'double' || t.startsWith('decimal')
      ) {
        const num = Number(val);
        if (Number.isFinite(num)) return String(num);
        // Fallback: cast string to the target numeric type
        return `CAST('${escapeSingle(val)}' AS ${t || 'DOUBLE'})`;
      }
      // default to string/varchar family
      return `'${escapeSingle(val)}'`;
    };

    // Build WHERE clause from partitions: { key: [v1,v2] } with proper typing
    const whereClauses = [];
    if (partitions && typeof partitions === 'object') {
      for (const [key, values] of Object.entries(partitions)) {
        if (Array.isArray(values) && values.length > 0) {
          const keyType = partitionKeyTypeByName[key] || 'string';
          const typedVals = values.map(v => formatLiteral(v, keyType)).join(', ');
          whereClauses.push(`"${key}" IN (${typedVals})`);
        }
      }
    }

    const whereSql = whereClauses.length ? ` WHERE ${whereClauses.join(' AND ')}` : '';
    const queryString = `SELECT * FROM "${database}"."${table}"${whereSql} LIMIT 10000`;

    const params = {
      QueryString: queryString,
      QueryExecutionContext: {
        Catalog: String(catalog),
        Database: String(database),
      },
      ResultConfiguration: {
        OutputLocation: outputLocation,
      },
      WorkGroup: process.env.ATHENA_WORKGROUP || undefined,
    };

    const out = await athena.send(new StartQueryExecutionCommand(params));
    const executionId = out.QueryExecutionId;
    if (!executionId) throw new Error('No QueryExecutionId returned');

    res.json({ executionId });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to start query execution' });
  }
});

app.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT} (region: ${region}, profile: ${process.env.AWS_PROFILE || 'default'})`);
});


